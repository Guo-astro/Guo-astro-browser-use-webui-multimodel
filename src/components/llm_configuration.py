import gradio as gr

def update_model_and_url(selected_provider):
    """Returns default model name and base URL based on the selected provider."""
    defaults = {
        "anthropic": {"model_name": "claude-3-5-haiku-latest", "base_url": "https://api.anthropic.com"},
        "openai": {"model_name": "gpt-3.5-turbo", "base_url": "https://api.openai.com"},
        "gemini": {"model_name": "gemini-2.0-flash-exp", "base_url": "https://api.gemini.com"},
        "deepseek": {"model_name": "deepseek-chat", "base_url": "https://api.deepseek.com"},
        "ollama": {"model_name": "llama3", "base_url": "http://localhost:11434/api/generate"},
    }
    provider_defaults = defaults.get(selected_provider, {"model_name": "", "base_url": ""})
    return provider_defaults["model_name"], provider_defaults["base_url"]

def create_llm_configuration_tab():
    """Creates and returns the LLM configuration tab"""
    with gr.TabItem("🔧 LLM Configuration", id="llm_configuration_tab"):
        with gr.Group():
            llm_provider = gr.Dropdown(
                ["anthropic", "openai", "gemini", "azure_openai", "deepseek", "ollama"],
                label="LLM Provider",
                value="deepseek",
                info="Select your preferred language model provider"
            )
            llm_model_name = gr.Textbox(
                label="Model Name",
                value="deepseek-chat",
                info="Specify the model to use"
            )
            llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Controls randomness in model outputs"
            )
            with gr.Row():
                llm_base_url = gr.Textbox(
                    label="Base URL",
                    value="https://api.deepseek.com",
                    info="API endpoint URL (if required)"
                )
                llm_api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    info="Your API key"
                )

            # Set up a change event on the provider dropdown to update model name and base URL
            llm_provider.change(
                fn=update_model_and_url,
                inputs=llm_provider,
                outputs=[llm_model_name, llm_base_url]
            )

    return {
        "llm_provider": llm_provider,
        "llm_model_name": llm_model_name,
        "llm_temperature": llm_temperature,
        "llm_base_url": llm_base_url,
        "llm_api_key": llm_api_key
    }
