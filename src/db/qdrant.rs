use std::env;
use anyhow::Result;
use qdrant_client::{
    qdrant::{PointStruct, SearchResponse, UpsertPointsBuilder},
    Qdrant,
};

use crate::shared::embedding::EmbeddingVector;

use super::embedded_chunk::EmbeddedChunk;


pub struct QdrantService {
    client: Qdrant,
    collection: String,
}

impl QdrantService {
    /// Creates a new QdrantService by establishing a connection to the Qdrant database.
    ///
    /// # Panics
    /// Panics if the required environment variables `QDRANT_SERVER` or `QDRANT_COLLECTION` are not set,
    /// or if a connection to the Qdrant database cannot be established.
    pub fn new() -> Self {
        let qdrant_server = env::var("QDRANT_SERVER")
            .expect("QDRANT_SERVER not defined");
        let collection = env::var("QDRANT_COLLECTION")
            .expect("QDRANT_COLLECTION not defined");

        let client = Qdrant::from_url(&qdrant_server)
            .build()
            .unwrap_or_else(|e| panic!("Can't establish Qdrant DB connection: {:#?}", e));

        Self {
            client,
            collection,
        }
    }

    /// Performs a vector search in the Qdrant database using the given embedding.
    ///
    /// # Parameters
    /// - `embedding`: The embedding tensor to be converted into a vector of f32 values.
    ///
    /// # Returns
    /// Returns a `Result` containing the search response from Qdrant if successful.
    ///
    /// # Errors
    /// Returns an error if the tensor conversion fails or if the Qdrant search query encounters issues.
    pub async fn vector_search(&self, embedding: EmbeddingVector) -> Result<SearchResponse> {
        let search_result = self.client.search_points(embedding).await?;
        Ok(search_result.into())
    }

    /// Inserts a collection of embedded chunks into the Qdrant database.
    ///
    /// # Parameters
    /// - `embedded_chunks`: A vector of embedded chunks to be inserted.
    ///
    /// # Returns
    /// Returns a `Result<()>` indicating success or failure.
    pub async fn insert_chunks(&self, embedded_chunks: Vec<EmbeddedChunk>) -> Result<()> {
        println!("Upserting to qdrant...");
        let points: Vec<PointStruct> = embedded_chunks.into_iter().map(|c| c.into()).collect();
        let _client = self.client
            .upsert_points(UpsertPointsBuilder::new(self.collection.clone(), points))
            .await?;
        Ok(())
    }
}
