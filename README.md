# Interactive EEG Signal Dashboard

This dashboard is an interactive tool for preprocessing and visualizing EEG data from CSV files.

## Getting Started

### Prerequisites

* **Python 3.10.0** is recommended. The application was developed and tested using this version.
* An account with [Supabase](https://supabase.com) for file storage.

### Installation

1.  Clone this repository or download the source code.

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  Create a `.env` file in the root directory and add your Supabase credentials:
    ```
    SUPABASE_URL="YOUR_SUPABASE_URL"
    SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
    ```

### Running the Application

To start the dashboard, run the following command in your terminal:
```bash
python dashboard2.py

### Note:
**
 The function "calculate_average_band_percentages" might not provide correct percentage calculations due to various factors like noise and artifacts that interfere even after applying standard EEG preprocessing... So this function is still under development and may produce incorrect percentages.**
