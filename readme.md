# FAISS Index Builder on Modal

This project builds FAISS indexes for product images using OpenAI's CLIP model and stores them in a Modal persisted volume. It fetches image data from a Supabase database, processes it, and generates three types of indexes: color, structure (grayscale), and combined.

## Project Structure

-   `build_index.py`: The main Python script containing the Modal application and the logic for building the FAISS indexes.
-   `.env.example`: A template for your environment variables.
-   `.gitignore`: Specifies intentionally untracked files to ignore.
-   `README.md`: This file.

## Prerequisites

Before you begin, ensure you have the following:

-   **Modal Account:** If you don't have one, sign up at [modal.com](https://modal.com/).
-   **Modal CLI:** Install the Modal CLI by running `pip install modal`.
-   **Supabase Project:** A Supabase project with a `product_images` table containing `id` and `image_url` columns.
-   **GitHub Repository:** A new, empty GitHub repository to host these files.

## Setup Instructions

1.  **Create a GitHub Repository:**
    * Go to GitHub and create a new repository (e.g., `modal-faiss-builder`).
    * Clone the repository to your local machine:
        ```bash
        git clone [https://github.com/YOUR_USERNAME/modal-faiss-builder.git](https://github.com/YOUR_USERNAME/modal-faiss-builder.git)
        cd modal-faiss-builder
        ```

2.  **Add Project Files:**
    * Create the `build_index.py` file and paste the provided Python code into it.
    * Create the `.env.example` file and paste the provided content into it.
    * Create the `.gitignore` file and paste the provided content into it.
    * Create the `README.md` file and paste this content into it.

3.  **Configure Supabase Secrets in Modal:**
    * Modal needs access to your Supabase credentials. **Do not commit your actual `.env` file to GitHub.**
    * Instead, create a Modal secret:
        ```bash
        modal secret create supabase-creds SUPABASE_URL="YOUR_SUPABASE_URL" SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
        ```
        Replace `"YOUR_SUPABASE_URL"` and `"YOUR_SUPABASE_ANON_KEY"` with your actual Supabase project URL and anon key. You can find these in your Supabase project settings under "API".

4.  **Commit and Push to GitHub:**
    * Add all the files to your Git repository:
        ```bash
        git add .
        ```
    * Commit the changes:
        ```bash
        git commit -m "Initial commit: Add FAISS index builder"
        ```
    * Push to GitHub:
        ```bash
        git push origin main
        ```

## Running the Application on Modal

Once your repository is set up and secrets are configured, you can deploy and run your Modal application.

1.  **Deploy the Modal App:**
    * From your terminal, in the root directory of your project, run:
        ```bash
        modal deploy build_index.py
        ```
    * This command will deploy your application to Modal. It will build the Docker image (if not already cached) and make your functions available.

2.  **Trigger the Index Building Job:**
    * Since the `if __name__ == "__main__":` block in `build_index.py` calls `build_all_indexes.remote()`, the index building job will automatically start when you deploy the application.
    * You can monitor the progress of the job in your Modal dashboard or by watching the logs in your terminal after deployment.

    * Alternatively, if you want to explicitly run the function after deployment (e.g., if you deployed without the `if __name__` block, or just want to re-run it), you can use:
        ```bash
        modal run build_index.py::app.build_all_indexes
        ```

## Persisted Volume

The `faiss-index-storage` volume is a persisted volume, meaning the FAISS indexes and the `progress.json` checkpoint file will be saved across runs. This allows the script to resume from where it left off if it's interrupted or rerun.

## Troubleshooting

-   **Supabase Credentials:** Double-check that your `SUPABASE_URL` and `SUPABASE_KEY` are correctly set in the Modal secret.
-   **Modal Deployment Issues:** If you encounter deployment errors, check the Modal CLI output for specific error messages. Ensure all necessary `pip_install` dependencies are listed.
-   **Image Processing Errors:** The script includes error handling for `UnidentifiedImageError` and other exceptions during image processing. Failed image URLs will be skipped.
-   **GPU Quota:** Ensure you have sufficient GPU quota on Modal if you are using the `A10G` GPU.
