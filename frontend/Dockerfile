FROM node:20

COPY package*.json .

RUN npm i

COPY . /app/

WORKDIR /app

CMD ["npm", "run", "dev"]
