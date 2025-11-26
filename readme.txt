cd backend
pip install -r requirements.txt
# run from repo root so artifacts/* paths resolve
uvicorn backend.main:app --host 0.0.0.0 --port 8000


cd frontend
npm i
ipconfig getifaddr en0  # get your Mac’s LAN IP
VITE_API_BASE=http://<LAN_IP>:8000 npm run dev
