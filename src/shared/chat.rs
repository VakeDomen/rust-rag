use ollama_rs::generation::chat::{request::ChatMessageRequest, ChatMessage};

#[derive(Debug, Clone)]
pub struct Chat {
    system_prompt: String,
    messages: Vec<ChatMessage>,
    model: String,
}

impl Chat {
    /// Creates a new Chat instance with a default system prompt and empty message history.
    pub fn new() -> Self {
        Self {
            system_prompt: "You are a helpful assistant.".to_owned(),
            messages: Vec::new(),
            model: "llama2:latest".to_owned(),
        }
    }
    
    /// Sets the model to be used.
    pub fn set_model(mut self, model: &str) -> Self {
        self.model = model.to_owned();
        self
    }
    
    /// Sets the system prompt.
    pub fn set_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.to_owned();
        self
    }
    
    /// Adds a user message to the chat history.
    pub fn add_user_message(mut self, message: &str) -> Self {
        self.messages.push(ChatMessage::user(message.to_owned()));
        self
    }
    
    /// Adds an assistant message to the chat history.
    pub fn add_assistant_message(mut self, message: &str) -> Self {
        self.messages.push(ChatMessage::assistant(message.to_owned()));
        self
    }
    
    /// Converts this Chat instance into a ChatMessageRequest.
    ///
    /// The system prompt is prepended as a system message and followed by the chat history.
    pub fn into_request(self) -> ChatMessageRequest {
        let mut full_messages = Vec::new();
        // Prepend the system prompt as a system message.
        full_messages.push(ChatMessage::system(self.system_prompt));
        // Append all other chat messages.
        full_messages.extend(self.messages);
        ChatMessageRequest::new(self.model, full_messages)
    }
}

impl From<Chat> for ChatMessageRequest {
    fn from(chat: Chat) -> Self {
        chat.into_request()
    }
}
