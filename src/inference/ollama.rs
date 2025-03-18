use std::env;
use ollama_rs::{
    error::OllamaError, 
    generation::{
        completion::{GenerationResponse, GenerationResponseStream}, 
        embeddings::{request::GenerateEmbeddingsRequest, GenerateEmbeddingsResponse}
    }, 
    Ollama
};

use crate::shared::{embedding::Embeddable, question::Question};


#[derive(Debug)]
pub struct OllamaService {
    ollama: Ollama,
}

impl Default for OllamaService {
    fn default() -> Self {
        let ollama_host = env::var("OLLAMA_HOST").expect("OLLAMA HOST not set");
        let ollama_port = env::var("OLLAMA_PORT").expect("OLLAMA PORT not set");
        let ollama_port: u16 = ollama_port.parse().expect("OLLAMA_PORT not u16");

        Self { 
            ollama: Ollama::new(ollama_host, ollama_port) 
        }
    }
}

impl OllamaService {

    pub fn new(host: String, port: u16) -> Self {
        Self { 
            ollama: Ollama::new(host, port) 
        }
    }

    pub async fn generate(&self, question: Question) -> Result<GenerationResponse, OllamaError> {
        self.ollama.generate(question.into()).await
    }

    pub async fn generate_stream(&self, question: Question) -> Result<GenerationResponseStream, OllamaError> {
        self.ollama.generate_stream(question.into()).await
    }

    pub async fn generate_all(&self, questions: Vec<Question>) -> Vec<Result<GenerationResponse, OllamaError>> {
        let futures = questions.into_iter().map(|q| async move {
            self.generate(q).await
        });
    
        let results = futures::future::join_all(futures).await;
        results.into_iter().collect()
    }

    pub async fn generate_and_parse<T: From<GenerationResponse>>(&self, questions: Question) -> anyhow::Result<T> {
        Ok(T::from(self.generate(questions).await?))
    }

    pub async fn generate_and_parse_all_results<T: From<GenerationResponse>>(&self, questions: Vec<Question>) -> Vec<anyhow::Result<T>> {
        let futures = questions.into_iter().map(|q| async move {
            self.generate_and_parse(q).await
        });
        futures::future::join_all(futures).await
    }

    pub async fn generate_and_parse_all<T: From<GenerationResponse>>(&self, questions: Vec<Question>) -> Vec<Option<T>> {
        let res = self.generate_and_parse_all_results(questions).await;
        res.into_iter().map(|r| r.ok()).collect()
    }





    pub async fn embed_req(&self, req: GenerateEmbeddingsRequest) -> Result<GenerateEmbeddingsResponse, OllamaError> {
        self.ollama.generate_embeddings(req).await
    }

    pub async fn embed(&self, emb: &dyn Embeddable) -> Result<GenerateEmbeddingsResponse, OllamaError> {
        let req: GenerateEmbeddingsRequest = emb.into_embed();
        self.embed_req(req).await
    }

    pub async fn embed_all(&self, embs: Vec<&dyn Embeddable>) -> Vec<Result<GenerateEmbeddingsResponse, OllamaError>> {
        
        let futures = embs.into_iter().map(|q| async move {
            let req: GenerateEmbeddingsRequest = q.into_embed();
            self.embed_req(req).await
        });

        let results = futures::future::join_all(futures).await;
        results.into_iter().collect()
    }

    pub async fn embed_and_parse<T: From<GenerateEmbeddingsResponse>>(&self, emb: &dyn Embeddable) -> anyhow::Result<T> {
        let req: GenerateEmbeddingsRequest = emb.into_embed();
        Ok(T::from(self.embed_req(req).await?))
    }

    pub async fn embed_and_parse_all_results<T: From<GenerateEmbeddingsResponse>>(&self, embs: Vec<&dyn Embeddable>) -> Vec<anyhow::Result<T>> {
        let futures = embs.into_iter().map(|q| async move {
            self.embed_and_parse(q).await
        });
        futures::future::join_all(futures).await
    }

    pub async fn embed_and_parse_all<T: From<GenerateEmbeddingsResponse>>(&self, embs: Vec<&dyn Embeddable>) -> Vec<Option<T>> {
        let res = self.embed_and_parse_all_results(embs).await;
        res.into_iter().map(|r| r.ok()).collect()
    }

}
 