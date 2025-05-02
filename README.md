# Axolotl Thinking Weight Plugin

This plugin extends the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) training framework to allow for differential weighting of loss contributions between "thinking" steps (enclosed in specific tags) and the final "answer" part of an assistant's response during training.

This is useful for training reasoning models, where you want the model to learn the reasoning process but prioritize the correctness of the final output.

## ❗ Important Prerequisite: Single-Token Tags

This plugin **critically assumes** that the start tag (default: `<think>`) and the end tag (default: `</think>`) are each tokenized as a **single, unique token ID** by your chosen tokenizer.

*   **Validation:** The plugin will check this during initialization. If a tag tokenizes into multiple IDs, the plugin will raise a `ValueError` and disable the weighted loss feature (falling back to standard loss).
*   **How to Fix:** If your tokenizer splits the tags, you **must** add them as single tokens. You can do this in your main Axolotl `config.yml` using (assuming they don't conflict with existing tokens):
```yaml
tokens:
  - "<think>"
  - "</think>"
```

## Installation

1.  **Install Axolotl:**
    Make sure you have a working installation of `axolotl`.
    ```bash
    pip install axolotl
    ```

2.  **Install this plugin:**
    You can install this plugin directly from its source repository.

    ```bash
    pip install git+https://github.com/nidhishs/axolotl-thinking-weight-plugin.git
    ```

## Configuration

Modify your Axolotl `config.yml` to use the plugin:

1.  **Register the Plugin:** Add the plugin's Python import path to the `plugins` list.

    ```yaml
    plugins:
      - axolotl_plugins.thinking_weight.ThinkingWeightPlugin
    ```

2.  **Configure Weights and Tags:** Add the following parameters to your config root.
    ```yaml
    # ... your other Axolotl config (base_model, datasets, etc.) ...

    # --- Thinking Weight Plugin Config ---
    use_thinking_weight: true      # Default: true (set to false to disable weighting even if plugin is active)
    thinking_token_weight: 0.6     # Default: 0.6 (Weight for tokens inside tags)
    answer_token_weight: 1.0       # Default: 1.0 (Weight for other trainable assistant tokens)
    think_start_token: "<think>"   # Default: "<think>" (Customize if needed)
    think_end_token: "</think>"    # Default: "</think>" (Customize if needed)
    ```

3.  **Configure Dataset:**
    Ensure your dataset includes the thinking tags (e.g., `<think>...</think>`) directly within the assistant's message `content`.