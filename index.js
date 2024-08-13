const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const width = canvas.width;
const height = canvas.height;

let numPredators = 4;
let numPrey = 16;
let numFood = 64;

let predators = [];
let prey = [];
let food = [];
let maxTimeAlive = 0;
let deadAgents = [];
let bestWeights = null;
let bestFitness = -Infinity;
let generation = 1;
const populationSize = 10;
let isRunning = false;
let bestPredators = [];
let bestPrey = [];

function random(x) {
  return Math.random() * x;
}

class Agent {
  constructor(type, x, y, model = null) {
    this.type = type;
    this.x = x || random(width);
    this.y = y || random(height);
    this.vx = random(2) - 1;
    this.vy = random(2) - 1;
    this.color = type === "prey" ? "green" : "red";
    this.size = type === "prey" ? 10 : 15;
    this.id = this.type === "prey" ? prey.length : predators.length;
    this.timeAlive = 0;
    this.health = type === "prey" ? 1000 : 500;
    this.model = model || this.createModel();
  }

  createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [this.inputs().length], units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 2, activation: 'sigmoid' }));
    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError' });
    return model;
  }

  averageWeights(model1, model2) {
    const weights1 = model1.getWeights();
    const weights2 = model2.getWeights();
    const averagedWeights = weights1.map((weight, index) => {
      return weight.add(weights2[index]).div(tf.scalar(2));
    });
    return averagedWeights;
  }

  checkBreeding() {
    const agentsList = this.type === "prey" ? prey : predators;
    agentsList.forEach(agent => {
      if (agent !== this && this.isColliding(agent)) {
        const averagedWeights = this.averageWeights(this.model, agent.model);
        const newAgent = new Agent(this.type, this.x, this.y, this.createModel());
        newAgent.model.setWeights(averagedWeights);

        if (this.type === "prey") {
          prey.push(newAgent);
        } else {
          predators.push(newAgent);
        }

        agent.die(); // Remove the agent that collided
        this.die(); // Remove the current agent as well
      }
    });
  }

  inputs() {
    const closest = this.closestSameType();
    const typeEncoding = this.type === "prey" ? [1, 0] : [0, 1];

    const inputsArray = [
      this.x / width,
      this.y / height,
      this.vx,
      this.vy,
      closest ? closest.x / width : 0,
      closest ? closest.y / height : 0,
      this.health / 500,
      maxTimeAlive > 0 ? this.timeAlive / maxTimeAlive : 0,
      canvas.width,
      canvas.height,
      ...typeEncoding
    ];

    food.forEach(f => {
      inputsArray.push(f.x / width, f.y / height);
    });

    predators.forEach(p => {
      inputsArray.push(p.x / width, p.y / height);
    });

    prey.forEach(pr => {
      inputsArray.push(pr.x / width, pr.y / height);
    });

    const maxFood = numFood; 
    const maxPredators = numPredators; 
    const maxPrey = numPrey; 
    
    const expectedLength = 12 + 2 * (maxFood + maxPredators + maxPrey);

    while (inputsArray.length < expectedLength) {
      inputsArray.push(0, 0);
    }

    return inputsArray;
  }

  predict(inputs) {
    const output = this.model.predict(tf.tensor([inputs])).dataSync();
    return output;
  }

  closestSameType() {
    let minDist = Infinity;
    let closest = null;
    const agentsList = this.type === "prey" ? prey : predators;
    agentsList.forEach(agent => {
      if (agent !== this) {
        const dist = Math.sqrt((this.x - agent.x) ** 2 + (this.y - agent.y) ** 2);
        if (dist < minDist) {
          minDist = dist;
          closest = agent;
        }
      }
    });
    return closest;
  }

  fitness() {
    const survivalFitness = this.timeAlive;
    const distanceFitness = 1 / (this.closestSameType() ? Math.sqrt((this.x - this.closestSameType().x) ** 2 + (this.y - this.closestSameType().y) ** 2) : 1);
    return survivalFitness + distanceFitness;
  }

  move() {
    this.x += this.vx;
    this.y += this.vy;
    if (this.x < 0 || this.x > width) this.vx *= -1;
    if (this.y < 0 || this.y > height) this.vy *= -1;
  }

  draw() {
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.fill();
  }

  die() {
    deadAgents.push(this);
    this.model.dispose();
    if (this.type === "prey") {
      prey = prey.filter(agent => agent !== this);
    } else {
      predators = predators.filter(agent => agent !== this);
    }
    // Check if population is zero
    if (prey.length === 0 && predators.length === 0) {
      resetPopulation();
    }
  }

  checkCollisions() {
    if (this.type === "predator") {
      prey.forEach(p => {
        if (this.isColliding(p)) {
          this.health += 1000;
          p.die();
        }
      });
    } else if (this.type === "prey") {
      food.forEach((f, index) => {
        if (this.isColliding(f)) {
          this.health += 500; 
          food[index].x = random(width);
          food[index].y = random(height);
          this.health += 1000;
        }
      });
    }
    //this.checkBreeding(); 
  }

  isColliding(other) {
    const dist = Math.sqrt((this.x - other.x) ** 2 + (this.y - other.y) ** 2);
    return dist < this.size + other.size;
  }

  update() {
    const inputs = this.inputs();
    const output = this.predict(inputs);
    this.vx += output[0] * 2 - 1;
    this.vy += output[1] * 2 - 1;
    this.move();
    this.checkCollisions();
    this.timeAlive++;
    if (this.type === "prey" && this.health <= 0) {
      this.die();
    }
    if (this.type === "predator" && this.health <= 0) {
      this.die();
    }
  }
}

function init() {
  predators = [];
  prey = [];
  food = [];
  deadAgents = [];
  for (let i = 0; i < numPredators; i++) {
    predators.push(new Agent("predator"));
  }
  for (let i = 0; i < numPrey; i++) {
    prey.push(new Agent("prey"));
  }
  for (let i = 0; i < numFood; i++) {
    food.push({ x: random(width), y: random(height) });
  }
}

function resetPopulation() {
  init();
  maxTimeAlive = 0;
  generation++;
  bestPredators = [];
  bestPrey = [];
}

function update() {
  predators.forEach(predator => predator.update());
  prey.forEach(p => p.update());
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  food.forEach(f => {
    ctx.fillStyle = "yellow";
    ctx.beginPath();
    ctx.arc(f.x, f.y, 5, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.fill();
  });
  predators.forEach(predator => predator.draw());
  prey.forEach(p => p.draw());
}

function gameLoop() {
  if (isRunning) {
    update();
    draw();
    requestAnimationFrame(gameLoop);
  }
}

document.getElementById("showSessionsBtn").addEventListener("click", () => {
  const sessionsList = document.getElementById("sessionsList");
  sessionsList.innerHTML = '';
  const savedSessions = JSON.parse(localStorage.getItem("savedSessions")) || [];
  savedSessions.forEach((session, index) => {
    const li = document.createElement("li");
    li.textContent = `Session ${index + 1}: Generation ${session.generation}, Best Fitness ${session.bestFitness}`;
    sessionsList.appendChild(li);
  });
});

document.getElementById("startBtn").addEventListener("click", () => {
  if (!isRunning) {
    isRunning = true;
    gameLoop();
  }
});

document.getElementById("pauseBtn").addEventListener("click", () => {
  isRunning = false;
});

document.getElementById("resetBtn").addEventListener("click", () => {
  resetPopulation();
});

init();

