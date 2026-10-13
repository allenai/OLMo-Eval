# Build context: dashboard/ (so the UI can import ../contract types).
FROM node:24-alpine AS build
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY contract/ /app/contract/
COPY ui/ /app/ui/
RUN npm run build && rm -f dist/mockServiceWorker.js

# Runs as an unprivileged user (uid 101) and listens on 8080.
FROM nginxinc/nginx-unprivileged:alpine
COPY --from=build /app/ui/dist /usr/share/nginx/html
COPY ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY ui/security-headers.conf /etc/nginx/snippets/security-headers.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
