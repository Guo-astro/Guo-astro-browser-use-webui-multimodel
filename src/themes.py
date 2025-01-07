"""Theme configuration for the Browser Use WebUI"""

from gradio.themes import Default, Soft, Monochrome, Glass, Origin, Citrus, Ocean

class ProductionTheme(Soft):
    """Custom theme extending the Soft theme with refined styling"""
    
    def __init__(self, primary_hue="blue", secondary_hue="orange", neutral_hue="gray"):
        super().__init__(primary_hue=primary_hue, secondary_hue=secondary_hue, neutral_hue=neutral_hue)
        self.set(
            body_background_fill="#f9fafb",
            panel_background_fill="white",
        )

THEME_MAP = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Production": ProductionTheme()
}

BASE_CSS = """
/* Container to be responsive, centered, with some padding */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding-top: 20px !important;
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}

/* Header text styling */
.header-text {
    text-align: center;
    margin-bottom: 30px;
}

/* Tab style improvements */
.gradio-container .tabs .tabitem {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 20px;
}

/* Smoother corners for group containers */
.gradio-container .gr-group {
    border-radius: 8px;
    background: #fefefe;
    padding: 16px;
    border: 1px solid #eaeaea;
}

/* Style the run and stop buttons */
button#run_button {
    background-color: #0b5ed7 !important;
    color: #fff !important;
    border: none !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease;
}

button#run_button:hover {
    background-color: #084ca1 !important;
}

button#stop_button {
    background-color: #dc3545 !important;
    color: #fff !important;
    border: none !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease;
}

button#stop_button:hover {
    background-color: #b52d36 !important;
}

/* Textboxes, checkboxes, radio buttons, etc. */
.gr-textbox textarea {
    background: #fafafa !important;
    border: 1px solid #ccc !important;
    border-radius: 6px !important;
    transition: border-color 0.3s ease;
}

.gr-textbox textarea:focus {
    outline: none !important;
    border-color: #888 !important;
}

/* Make label texts slightly bolder and spaced */
label {
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}

/* A subtle hover effect for textual inputs */
.gr-textbox:hover textarea {
    border-color: #bbb !important;
}
"""

BASE_JS = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""