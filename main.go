package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	openai "github.com/sashabaranov/go-openai"
)

type ChatRequest struct {
	Model       string                         `json:"model"`
	Messages    []openai.ChatCompletionMessage `json:"messages"`
	Temperature float32                        `json:"temperature"`
	TopP        float32                        `json:"top_p"`
	MaxTokens   int                            `json:"max_tokens"`
	Stream      bool                           `json:"stream"`

	// Дополнительные поля, которые просто проглатываем
	KeepAlive json.RawMessage `json:"keep_alive"`
	Format    json.RawMessage `json:"format"`
	Tools     json.RawMessage `json:"tools"`
	Options   json.RawMessage `json:"options"`
}

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENROUTER_API_KEY не задан")
	}

	// Настраиваем клиента с нужным BaseURL
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://openrouter.ai/api/v1"
	client := openai.NewClientWithConfig(config)

	r := gin.Default()

	// Обработчик для списка моделей
	modelsHandler := func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"object": "list",
			"data": []gin.H{
				{
					"id":                    "deepseek/deepseek-chat-v3-0324:free",
					"object":                "model",
					"type":                  "llm",
					"publisher":             "openrouter",
					"arch":                  "llama",
					"compatibility_type":    "openai",
					"quantization":          "none",
					"state":                 "loaded",
					"max_context_length":    16384,
					"loaded_context_length": 16384,
					"created":               time.Now().Unix(),
				},
			},
		})
	}

	r.GET("/v1/models", modelsHandler)
	r.GET("/api/v0/models", modelsHandler)

	// Обработчик для потоковых чатов
	chatHandler := func(c *gin.Context) {
		var req ChatRequest
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
			return
		}

		log.Printf("[PROMPT] model=%s, messages=%d", req.Model, len(req.Messages))
		// Если приходит название модели в старом формате — меняем
		if req.Model == "deepseek-r1-distill-llama-8b" {
			req.Model = "deepseek/deepseek-chat-v3-0324:free"
		}
		if req.MaxTokens <= 0 {
			req.MaxTokens = 1024
		}

		opts := openai.ChatCompletionRequest{
			Model:       req.Model,
			Messages:    req.Messages,
			Temperature: req.Temperature,
			TopP:        req.TopP,
			MaxTokens:   req.MaxTokens,
			Stream:      req.Stream,
		}

		// Контекст с увеличенным таймаутом
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		stream, err := client.CreateChatCompletionStream(ctx, opts)
		if err != nil {
			log.Printf("[ERROR] %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer stream.Close()

		// Устанавливаем заголовки для SSE
		c.Writer.Header().Set("Content-Type", "text/event-stream")
		c.Writer.Header().Set("Cache-Control", "no-cache")
		c.Writer.Header().Set("Connection", "keep-alive")
		c.Writer.Header().Set("X-Accel-Buffering", "no")
		c.Writer.WriteHeader(http.StatusOK)
		c.Writer.Flush()

		var lastChunk []byte = nil

		// Обрабатываем и отправляем чанки
		for {
			resp, err := stream.Recv()
			if err != nil {
				log.Printf("[STREAM END] err: %v", err)
				// Если есть буферизированный последний чан, обрабатываем его
				if lastChunk != nil {
					var respMap map[string]interface{}
					if err := json.Unmarshal(lastChunk, &respMap); err == nil {
						if choices, ok := respMap["choices"].([]interface{}); ok && len(choices) > 0 {
							if choiceMap, ok := choices[0].(map[string]interface{}); ok {
								if delta, ok := choiceMap["delta"].(map[string]interface{}); ok {
									// Если нет ключа "content", значит это финальный чан
									if _, hasContent := delta["content"]; !hasContent {
										choiceMap["delta"] = map[string]interface{}{}
										choiceMap["finish_reason"] = "stop"
									}
								}
							}
						}
						// Пересериализуем последний чан
						lastChunk, _ = json.Marshal(respMap)
					}
					// Отправляем буферизированный последний чан
					fmt.Fprintf(c.Writer, "data: %s\n\n", lastChunk)
					c.Writer.Flush()
				}
				// Отправляем финальную строку [DONE]
				fmt.Fprintf(c.Writer, "data: [DONE]\n\n")
				c.Writer.Flush()
				break
			}

			// Если уже есть буферизированный чан, отправляем его
			if lastChunk != nil {
				fmt.Fprintf(c.Writer, "data: %s\n\n", lastChunk)
				c.Writer.Flush()
			}

			// Получаем JSON-байты ответа для текущего чанка
			jsonBytes, err := json.Marshal(resp)
			if err != nil {
				log.Printf("[JSON ERROR] %v", err)
				continue
			}

			// Приводим формат к тому, что возвращает LM Studio, устанавливая logprobs в null
			var respMap map[string]interface{}
			if err := json.Unmarshal(jsonBytes, &respMap); err == nil {
				if choices, ok := respMap["choices"].([]interface{}); ok {
					for _, choice := range choices {
						if choiceMap, ok := choice.(map[string]interface{}); ok {
							choiceMap["logprobs"] = nil
						}
					}
					jsonBytes, _ = json.Marshal(respMap)
				} else {
					log.Printf("[UNMARSHAL ERROR] %v", err)
				}
			}

			// Буферизуем текущий чанк
			lastChunk = jsonBytes
		}
	}

	r.POST("/v1/chat/completions", chatHandler)
	r.POST("/api/v0/chat/completions", chatHandler)

	r.Run("127.0.0.1:1234")
}
