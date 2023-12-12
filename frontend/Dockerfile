# FROM node:21-alpine as build


FROM node:21-alpine

WORKDIR /app

COPY package*.json .

RUN npm i

COPY . .

CMD [ "npm", "run", "dev" ]
