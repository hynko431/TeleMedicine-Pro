# TeleMedicine Pro

Robust telemedicine platform with a FastAPI backend and a React (CRA + CRACO) frontend.

## Quick Start

1) Clone the repo and create env files from examples
2) Start backend (FastAPI + Uvicorn)
3) Start frontend (React)

### Backend
- Requirements: Python 3.11+, MongoDB
- Create your env file

````bash
cp backend/.env.example backend/.env
````

- Install dependencies and run the API

````bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
uvicorn backend.server:app --reload --port 8000
````

- Default important envs (see backend/.env.example):
  - MONGO_URL, DB_NAME
  - CORS_ORIGINS (e.g., http://localhost:3000)
  - EMERGENT_LLM_KEY

### Frontend
- Requirements: Node 18+ / 20+
- Create your env file

````bash
cp frontend/.env.example frontend/.env
````

- Point the app to the backend
  - REACT_APP_API_BASE_URL=http://localhost:8000

- Install and run (if package.json exists in frontend/)

````bash
cd frontend
npm ci
npm start
````

## Tests and CI
- Backend: CI checks Python formatting (black --check) and compiles sources to catch syntax errors
- Frontend: CI installs and builds if a package.json is present

Workflow file: .github/workflows/ci.yml

## Security and Secrets
- .env files are ignored by Git and not tracked
- Use backend/.env.example and frontend/.env.example as templates

## Branch Protection and Repo Settings (recommended)
- Protect main branch and require CI to pass before merging:
  - GitHub → Settings → Branches → Add rule for main → Require status checks
- Enable Issues/Projects:
  - GitHub → Settings → General → Features → Enable Issues and Projects

## Project Structure (high-level)
- backend/ → FastAPI app (server.py), requirements.txt, .env.example
- frontend/ → CRA app (CRACO config), .env.example
- .github/workflows/ci.yml → CI pipeline

## License
See LICENSE in the repository.
