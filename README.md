# Looper Dashboard

This repository contains a dashboarding tool for processing and visualizing machine data from the "Looper" foam production machine. It consists of two core components:

## 1. **Live Monitoring**
- Displays critical machine data in real-time (Aggregate Height, Main Conveyor Belt Speed)
- Machine state classification is performed live using ML algorithms.
- Data is fetched from the database every second and classified live.

## 2. **Batch Analysis**
- Allows selecting a specific timeframe for analysis.
- **Note**: Currently, only the date **12.11.24** is supported due to the development phase.
- Calculates the **Overall Equipment Efficiency (OEE)** for the selected timeframe.

---

## Tech Stack
The dashboard is built using:
- **[Dash](OEE_DB/dash/)**: Frontend for data visualization and interaction.
- **[FastAPI](OEE_DB/fastapi/)**: Backend for serving the data and managing API endpoints.
- **[PostgreSQL](OEE_DB/postgres_with_data/)**: Database for storing and fetching machine data.
- **Docker**: Containerization for consistent deployment and reuse.

---

## Prerequisites
- Ensure **Docker Desktop** is installed on your machine. If not, follow the installation guide: [Docker Get Started](https://www.docker.com/get-started/).
- Recommended browser: **Google Chrome** for optimal performance.

---

## Setup / Getting Started
Follow these steps to run the dashboard on your local machine:

### Step 1: Clone the Repository
```bash
git clone https://github.com/sayyamalam/looper-dashboard.git
```

### Step 2: Start Docker Desktop
- Open Docker Desktop on your machine. 
- If you don't have it installed, download and install it from [Docker](https://www.docker.com/get-started/).

### Step 3: Navigate to the `OEE_DB` Directory
Change into the `OEE_DB` folder inside the cloned repository:
```bash
cd looper-dashboard/OEE_DB
```

### Step 4: Run Docker Compose
Run the following command to start the application:
```bash
docker compose up
```

### Step 5: Open the Dashboard in Your Browser
Open your browser and navigate to: http://localhost:8050
Note: The tool is optimized for Google Chrome, and we recommend using it for the best experience.

