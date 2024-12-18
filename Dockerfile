
# Stage 1: Build frontend
FROM node:16 AS frontend-builder
WORKDIR /frontend
COPY ./chat-front .
RUN npm install && npm run build

# Stage 2: Set up the Python environment for FastAPI
FROM python:3.9

# Set the working directory for the backend
WORKDIR /app

# Copy and install backend dependencies
COPY ./chat_back/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY ./chat_back .

RUN pip install chromadb

# Copy the built frontend from the previous stage
COPY --from=frontend-builder /frontend/dist ./dist

# Serve the static files from the current working directory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]



