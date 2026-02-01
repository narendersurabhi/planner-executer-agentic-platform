FROM node:20-alpine
WORKDIR /app
COPY services/ui/package.json /app/package.json
RUN npm install
COPY services/ui /app
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
