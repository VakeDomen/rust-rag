use std::env;

use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use anyhow::Result;
use qdrant_client::qdrant::SearchPoints;

use crate::db::embedded_chunk::EmbeddedChunk;


#[derive(Debug, Clone)]
pub struct EmbeddingVector(pub Vec<f32>);

pub trait Embeddable {
    fn into_embed(&self) -> GenerateEmbeddingsRequest;
    fn set_embedding_vectors(&mut self, embedding_vector: Vec<EmbeddingVector>);
    fn prepare_for_upload(self, parent_doc_id: String, doc_summary: Option<String>) -> Result<Vec<EmbeddedChunk>>;
}


impl Into<SearchPoints> for EmbeddingVector {
    fn into(self) -> SearchPoints {
        let qdrant_collection = env::var("QDRANT_COLLECTION").expect("QDRANT_COLLECTION not defined");
        SearchPoints { 
            collection_name: qdrant_collection, 
            vector: self.0, 
            limit: 10,
            with_payload: Some(true.into()),
            with_vectors: Some(false.into()),
            ..Default::default()
        }
    }
}
