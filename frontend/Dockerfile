# Imagen del frontend: compila con Vite y sirve con Nginx, con proxy a la API.
FROM node:22-alpine AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* /app/
RUN npm install --no-audit --no-fund

COPY frontend /app
RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
