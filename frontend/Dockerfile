FROM node:21

COPY package*.json .

RUN npm i

COPY . /app/

WORKDIR /app

CMD ["npm", "run", "dev"]
