// backend/main.go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	uploadDir = "./uploads"
	outputDir = "./img"
)

func init() {
	os.MkdirAll(uploadDir, os.ModePerm)
	os.MkdirAll(outputDir, os.ModePerm)
}

type UploadResponse struct {
	Filename string `json:"filename"`
}

func handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	file, handler, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "No file uploaded", http.StatusBadRequest)
		return
	}
	defer file.Close()

	ext := strings.ToLower(filepath.Ext(handler.Filename))
	if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
		http.Error(w, "Only JPG/PNG images are allowed", http.StatusBadRequest)
		return
	}

	timestamp := time.Now().UnixNano()
	inputPath := filepath.Join(uploadDir, fmt.Sprintf("input_%d%s", timestamp, ext))
	outputFilename := fmt.Sprintf("result_%d.png", timestamp)
	outputPath := filepath.Join(outputDir, outputFilename)

	outFile, err := os.Create(inputPath)
	if err != nil {
		log.Printf("Failed to create upload file: %v", err)
		http.Error(w, "Server error", http.StatusInternalServerError)
		return
	}
	defer outFile.Close()

	_, err = outFile.ReadFrom(file)
	if err != nil {
		log.Printf("Failed to write upload file: %v", err)
		http.Error(w, "Server error", http.StatusInternalServerError)
		return
	}

	// Jalankan Python middleware (coba python3, fallback ke python)
	var stderr bytes.Buffer
	py := "python3"
	if _, err := exec.LookPath(py); err != nil {
		py = "python"
	}
	cmd := exec.Command(py, "middleware/pixel_art.py", inputPath, outputPath)
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		log.Printf("Python failed: %v", err)
		log.Printf("Python stderr: %s", stderr.String())
		http.Error(w, "Image processing failed", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(UploadResponse{Filename: outputFilename})
}

func handleResult(w http.ResponseWriter, r *http.Request) {
	filename := filepath.Base(r.URL.Path[len("/results/"):])
	if !strings.HasSuffix(filename, ".png") {
		http.Error(w, "Invalid file type", http.StatusBadRequest)
		return
	}

	fullPath := filepath.Join(outputDir, filename)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		http.Error(w, "File not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "image/png")
	http.ServeFile(w, r, fullPath)
}

func main() {
	// Serve static assets from the sibling frontend folder
	http.Handle("/assets/", http.StripPrefix("/assets/", http.FileServer(http.Dir("../frontend/assets"))))
	// Serve favicon if present; otherwise respond with 204 No Content to avoid 404 noise
	http.HandleFunc("/favicon.ico", func(w http.ResponseWriter, r *http.Request) {
		fav := "../frontend/favicon.ico"
		if _, err := os.Stat(fav); os.IsNotExist(err) {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		http.ServeFile(w, r, fav)
	})
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "../frontend/index.html")
	})
	http.HandleFunc("/upload", handleUpload)
	http.HandleFunc("/results/", handleResult)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	fmt.Printf("✅ Server running on http://localhost:%s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
