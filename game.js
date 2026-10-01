'use strict';

const GAME = {
  width: 400,
  height: 600,
  ground: 80,
  birdX: 90,
  birdRadius: 12,
  pipeWidth: 52,
  topMargin: 70,
  bottomMargin: 70
};

const PHYSICS = {
  gravity: 0.7,
  flap: -8,
  maxFall: 10
};

const DIFFICULTIES = {
  EASY: { speed: 2.4, gap: 170, spawn: 240 },
  HARD: { speed: 3.6, gap: 132, spawn: 200 },
  INSANE: { speed: 4.2, gap: 118, spawn: 185 }
};

const AI_CONFIG = {
  alpha: 0.2,
  gamma: 0.95,
  epsilon: 0.1,
  epsilonDecay: 0.999,
  epsilonMin: 0.001,
  exploreFlapProb: 0.1
};

const REWARDS = {
  alive: 1,
  pass: 10,
  crash: -100
};

const TRAIN_SPEEDS = [1, 5, 10, 25, 50];

const clamp = (value, min, max) =>
  Math.min(max, Math.max(min, value));

const mean = values =>
  values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;

const store = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },

  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch {
      return false;
    }
  },

  raw(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },

  setRaw(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {}
  },

  remove(key) {
    try {
      localStorage.removeItem(key);
    } catch {}
  }
};

function encodeState(y, velocityY, pipe) {
  const dx = pipe
    ? Math.floor(clamp(
        pipe.x + GAME.pipeWidth - GAME.birdX,
        0,
        240
      ) / 20)
    : 12;

  const dy = pipe
    ? Math.floor(clamp(
        pipe.gapY - y,
        -200,
        200
      ) / 10)
    : 0;

  const velocity = Math.round(
    clamp(velocityY, -10, 10) / 2
  );

  return `${dx}|${dy}|${velocity}`;
}

function computeReward(passed, crashed) {
  if (crashed) return REWARDS.crash;
  return REWARDS.alive + (passed ? REWARDS.pass : 0);
}

class QLearningAgent {
  constructor(config, random = Math.random) {
    this.config = config;
    this.random = random;
    this.q = new Map();
    this.epsilon = config.epsilon;
  }

  getRow(state) {
    if (!this.q.has(state)) {
      this.q.set(state, [0, 0]);
    }

    return this.q.get(state);
  }

  chooseAction(state, greedy = false) {
    if (!greedy && this.random() < this.epsilon) {
      return this.random() < this.config.exploreFlapProb ? 1 : 0;
    }

    const row = this.getRow(state);
    return row[1] > row[0] ? 1 : 0;
  }

  learn(state, action, reward, nextState, done) {
    const row = this.getRow(state);

    const future = done
      ? 0
      : Math.max(...this.getRow(nextState));

    row[action] += this.config.alpha * (
      reward +
      this.config.gamma * future -
      row[action]
    );
  }

  endEpisode() {
    this.epsilon = Math.max(
      this.config.epsilonMin,
      this.epsilon * this.config.epsilonDecay
    );
  }

  reset() {
    this.q.clear();
    this.epsilon = this.config.epsilon;
  }

  save() {
    return JSON.stringify({
      version: 1,
      epsilon: this.epsilon,
      q: [...this.q.entries()]
    });
  }

  load(json) {
    try {
      const data = JSON.parse(json);

      if (
        data.version !== 1 ||
        !Array.isArray(data.q) ||
        typeof data.epsilon !== 'number'
      ) {
        return false;
      }

      const map = new Map();

      for (const [key, values] of data.q) {
        if (
          typeof key !== 'string' ||
          !Array.isArray(values) ||
          values.length !== 2 ||
          !values.every(Number.isFinite)
        ) {
          return false;
        }

        map.set(key, [values[0], values[1]]);
      }

      this.q = map;
      this.epsilon = data.epsilon;

      return true;
    } catch {
      return false;
    }
  }
}

function maxGapShift(profile, skill) {
  const ticks = profile.spawn / profile.speed;

  return Math.min(
    220,
    Math.max(60, ticks * 1.5)
  ) * (0.6 + 0.4 * skill);
}

function nextGap(previousY, profile, skill, random = Math.random) {
  const low = GAME.topMargin + profile.gap / 2;

  const high =
    GAME.height -
    GAME.ground -
    GAME.bottomMargin -
    profile.gap / 2;

  return clamp(
    previousY +
    (random() * 2 - 1) * maxGapShift(profile, skill),
    low,
    high
  );
}

class DifficultyManager {
  constructor() {
    this.runs = [];
    this.skill = 0.3;
  }

  recordRun(score) {
    this.runs.push(score);

    if (this.runs.length > 10) {
      this.runs.shift();
    }

    this.skill = clamp(
      this.skill +
      (this.calculateSkill() - this.skill) * 0.25,
      0,
      1
    );
  }

  calculateSkill() {
    if (!this.runs.length) return 0.3;

    const average = mean(this.runs);

    const standardDeviation = Math.sqrt(
      mean(this.runs.map(score => (score - average) ** 2))
    );

    const half = Math.floor(this.runs.length / 2);

    const progress = half
      ? clamp(
          0.5 +
          (
            mean(this.runs.slice(half)) -
            mean(this.runs.slice(0, half))
          ) / (average + 5),
          0,
          1
        )
      : 0.5;

    const survival = clamp(average / 30, 0, 1);

    const consistency = clamp(
      1 - standardDeviation / (average + 1),
      0,
      1
    );

    const recent = clamp(
      this.runs[this.runs.length - 1] / 30,
      0,
      1
    );

    return clamp(
      0.4 * survival +
      0.3 * consistency +
      0.2 * progress +
      0.1 * recent,
      0,
      1
    );
  }

  getParams() {
    if (this.skill < 0.5) return DIFFICULTIES.EASY;
    if (this.skill < 0.85) return DIFFICULTIES.HARD;
    return DIFFICULTIES.INSANE;
  }

  getLabel() {
    const params = this.getParams();

    return Object.entries(DIFFICULTIES)
      .find(([, value]) => value === params)?.[0] || 'EASY';
  }
}

class FlappySimulation {
  constructor(params, skill = 0.5) {
    this.params = params;
    this.skill = skill;
    this.reset();
  }

  reset() {
    this.y = GAME.height / 2;
    this.velocityY = 0;
    this.pipes = [];
    this.score = 0;
    this.ticks = 0;
    this.alive = true;
    this.lastGapY = GAME.height / 2;
  }

  getNextPipe() {
    return this.pipes.find(
      pipe =>
        pipe.x + GAME.pipeWidth >
        GAME.birdX - GAME.birdRadius
    );
  }

  step(flap) {
    let passed = false;

    if (flap) {
      this.velocityY = PHYSICS.flap;
    }

    this.velocityY = Math.min(
      PHYSICS.maxFall,
      this.velocityY + PHYSICS.gravity
    );

    this.y += this.velocityY;
    this.ticks++;

    const lastPipe = this.pipes[this.pipes.length - 1];

    if (
      !lastPipe ||
      lastPipe.x < GAME.width - this.params.spawn
    ) {
      this.lastGapY = nextGap(
        this.lastGapY,
        this.params,
        this.skill
      );

      this.pipes.push({
        x: GAME.width,
        gapY: this.lastGapY,
        gap: this.params.gap,
        passed: false
      });
    }

    for (const pipe of this.pipes) {
      pipe.x -= this.params.speed;

      if (
        !pipe.passed &&
        pipe.x + GAME.pipeWidth < GAME.birdX
      ) {
        pipe.passed = true;
        this.score++;
        passed = true;
      }
    }

    if (
      this.pipes.length &&
      this.pipes[0].x + GAME.pipeWidth < 0
    ) {
      this.pipes.shift();
    }

    let crashed =
      this.y - GAME.birdRadius < 0 ||
      this.y + GAME.birdRadius > GAME.height - GAME.ground;

    for (const pipe of this.pipes) {
      const overlapsX =
        GAME.birdX + GAME.birdRadius > pipe.x &&
        GAME.birdX - GAME.birdRadius < pipe.x + GAME.pipeWidth;

      const outsideGap =
        this.y - GAME.birdRadius < pipe.gapY - pipe.gap / 2 ||
        this.y + GAME.birdRadius > pipe.gapY + pipe.gap / 2;

      if (overlapsX && outsideGap) {
        crashed = true;
      }
    }

    if (crashed) {
      this.alive = false;
    }

    return { passed, crashed };
  }
}

class CyberFlightScene extends Phaser.Scene {
  constructor() {
    super('CyberFlightScene');
  }

  create() {
    this.state = 'MENU';
    this.mode = 'PLAY';

    this.selectedDifficulty =
      store.get('cyber.difficulty', 'EASY');

    if (!DIFFICULTIES[this.selectedDifficulty]) {
      this.selectedDifficulty = 'EASY';
    }

    this.sim = new FlappySimulation(
      DIFFICULTIES[this.selectedDifficulty]
    );

    this.difficulty = new DifficultyManager();
    this.agent = new QLearningAgent(AI_CONFIG);

    this.highScore =
      Number(store.get('cyber.high', 0)) || 0;

    const savedModel = store.raw('cyber.model');

    if (savedModel && !this.agent.load(savedModel)) {
      store.remove('cyber.model');
    }

    this.flapQueued = false;
    this.trainSpeedIndex = 2;
    this.episode = 0;
    this.trainingBest = 0;
    this.recentScores = [];
    this.debug = false;

    this.graphics = this.add.graphics();

    this.setupInput();
    this.setSceneState('MENU');
  }

  setupInput() {
    this.input.keyboard.on(
      'keydown-SPACE',
      () => this.flap()
    );

    this.input.keyboard.on('keydown-P', () => {
      if (this.state === 'RUN') {
        this.state = 'PAUSED';
      } else if (this.state === 'PAUSED') {
        this.state = 'RUN';
      }
    });

    this.input.keyboard.on('keydown-R', () => {
      if (this.state !== 'MENU') {
        this.startGame(this.mode);
      }
    });

    this.input.keyboard.on('keydown-ESC', () => {
      this.setSceneState('MENU');
      window.GameUI.showMenu();
    });

    this.input.keyboard.on('keydown-D', () => {
      this.debug = !this.debug;
    });

    this.input.keyboard.on('keydown-PLUS', () => {
      this.trainSpeedIndex = Math.min(
        TRAIN_SPEEDS.length - 1,
        this.trainSpeedIndex + 1
      );
    });

    this.input.keyboard.on('keydown-MINUS', () => {
      this.trainSpeedIndex = Math.max(
        0,
        this.trainSpeedIndex - 1
      );
    });

    this.input.on('pointerdown', () => this.flap());
  }

  setSceneState(state) {
    this.state = state;
  }

  setDifficulty(name) {
    if (!DIFFICULTIES[name]) return;

    this.selectedDifficulty = name;
    store.set('cyber.difficulty', name);
    this.sim.params = DIFFICULTIES[name];
  }

  startGame(mode = 'PLAY') {
    this.mode = mode;
    this.state = 'RUN';

    let params;

    if (mode === 'PLAY') {
      params = DIFFICULTIES[this.selectedDifficulty];
    } else {
      params = DIFFICULTIES.HARD;
    }

    this.sim.params = params;
    this.sim.skill = this.difficulty.skill;
    this.sim.reset();
    this.flapQueued = false;

    window.GameUI.showGame();
  }

  flap() {
    if (this.state === 'RUN' && this.mode === 'PLAY') {
      this.flapQueued = true;
    }
  }

  update() {
    if (this.state === 'RUN') {
      const steps =
        this.mode === 'TRAIN'
          ? TRAIN_SPEEDS[this.trainSpeedIndex]
          : 1;

      for (
        let i = 0;
        i < steps && this.state === 'RUN';
        i++
      ) {
        this.tick();
      }
    }

    this.renderGame();

    window.GameUI.updateHUD(
      this.sim.score,
      this.highScore
    );
  }

  tick() {
    const simulation = this.sim;

    const stateBefore = encodeState(
      simulation.y,
      simulation.velocityY,
      simulation.getNextPipe()
    );

    const action =
      this.mode === 'PLAY'
        ? (this.flapQueued ? 1 : 0)
        : this.agent.chooseAction(
            stateBefore,
            this.mode === 'AI'
          );

    this.flapQueued = false;

    const result = simulation.step(action === 1);

    if (this.mode === 'TRAIN') {
      this.agent.learn(
        stateBefore,
        action,
        computeReward(result.passed, result.crashed),
        encodeState(
          simulation.y,
          simulation.velocityY,
          simulation.getNextPipe()
        ),
        result.crashed
      );
    }

    if (
      this.mode === 'TRAIN' &&
      simulation.score >= 500
    ) {
      simulation.alive = false;
    }

    if (simulation.alive) return;

    this.finishEpisode();
  }

  finishEpisode() {
    const score = this.sim.score;

    this.trainingBest = Math.max(this.trainingBest, score);

    if (this.mode === 'PLAY') {
      this.difficulty.recordRun(score);

      if (score > this.highScore) {
        this.highScore = score;
        store.set('cyber.high', score);
      }

      this.state = 'OVER';

      window.GameUI.showGameOver(
        score,
        this.highScore
      );

      return;
    }

    if (this.mode === 'TRAIN') {
      this.episode++;

      this.recentScores.push(score);

      if (this.recentScores.length > 100) {
        this.recentScores.shift();
      }

      this.agent.endEpisode();

      if (this.episode % 50 === 0) {
        store.setRaw(
          'cyber.model',
          this.agent.save()
        );
      }

      this.sim.reset();
      return;
    }

    this.sim.reset();
  }

  renderGame() {
    const g = this.graphics;
    const s = this.sim;

    g.clear();

    g.fillStyle(0x03050d, 1);
    g.fillRect(0, 0, GAME.width, GAME.height);

    g.lineStyle(1, 0x10253a, 0.55);

    for (let x = 0; x <= GAME.width; x += 40) {
      g.lineBetween(x, 0, x, GAME.height);
    }

    for (let y = 0; y <= GAME.height; y += 40) {
      g.lineBetween(0, y, GAME.width, y);
    }

    for (let i = 0; i < 12; i++) {
      const x =
        ((i * 83 - s.ticks * 0.8) %
          (GAME.width + 80) +
          GAME.width + 80) %
        (GAME.width + 80) - 40;

      const y =
        40 + (i * 47) % (GAME.height - GAME.ground - 80);

      g.fillStyle(0x00e5ff, 0.22);
      g.fillCircle(x, y, i % 3 === 0 ? 2 : 1);
    }

    for (const pipe of s.pipes) {
      const topHeight = pipe.gapY - pipe.gap / 2;
      const bottomY = pipe.gapY + pipe.gap / 2;

      g.fillStyle(0x00e5ff, 0.06);
      g.fillRect(pipe.x - 5, 0, GAME.pipeWidth + 10, topHeight);
      g.fillRect(
        pipe.x - 5,
        bottomY,
        GAME.pipeWidth + 10,
        GAME.height - bottomY
      );

      g.fillStyle(0x087f91, 1);
      g.fillRect(pipe.x, 0, GAME.pipeWidth, topHeight);
      g.fillRect(
        pipe.x,
        bottomY,
        GAME.pipeWidth,
        GAME.height - bottomY
      );

      g.fillStyle(0x00d9ed, 1);
      g.fillRect(pipe.x - 3, topHeight - 10, GAME.pipeWidth + 6, 10);
      g.fillRect(pipe.x - 3, bottomY, GAME.pipeWidth + 6, 10);
    }

    // Ground
    g.fillStyle(0x050812, 1);
    g.fillRect(
      0,
      GAME.height - GAME.ground,
      GAME.width,
      GAME.ground
    );

    g.lineStyle(2, 0x00e5ff, 0.7);
    g.lineBetween(
      0,
      GAME.height - GAME.ground,
      GAME.width,
      GAME.height - GAME.ground
    );

    const bird = this.add.container(GAME.birdX, s.y);
    const birdGraphics = this.add.graphics();

    bird.add(birdGraphics);

    birdGraphics.fillStyle(0x00e5ff, 0.09);
    birdGraphics.fillCircle(0, 0, GAME.birdRadius + 9);

    birdGraphics.fillStyle(0x00e5ff, 1);
    birdGraphics.fillCircle(0, 0, GAME.birdRadius + 2);

    birdGraphics.fillStyle(0x07131d, 1);
    birdGraphics.fillCircle(-2, 1, GAME.birdRadius - 3);

    birdGraphics.fillStyle(0xffffff, 1);
    birdGraphics.fillCircle(6, -5, 4);

    birdGraphics.fillStyle(0x03050d, 1);
    birdGraphics.fillCircle(7, -5, 2);

    birdGraphics.fillStyle(0xd946ef, 1);
    birdGraphics.fillTriangle(10, 0, 22, 4, 10, 8);

    birdGraphics.fillStyle(0x009fb4, 1);
    birdGraphics.fillEllipse(
      -5,
      4 + Math.sin(s.ticks / 3) * 3,
      13,
      8
    );

    bird.rotation = clamp(s.velocityY / 12, -0.5, 1.2);

    this.time.delayedCall(0, () => {
      bird.destroy();
    });

    if (this.debug && s.getNextPipe()) {
      const pipe = s.getNextPipe();

      g.lineStyle(1, 0xd946ef, 0.8);
      g.lineBetween(
        GAME.birdX,
        s.y,
        pipe.x,
        pipe.gapY
      );
    }
  }
}

window.GameUI = {
  selectedDifficulty: 'EASY',
  elements: {},

  init() {
    this.elements = {
      menu: document.querySelector('#menu-screen'),
      gameOver: document.querySelector('#game-over-screen'),
      hud: document.querySelector('#hud'),
      devPanel: document.querySelector('#dev-panel'),

      play: document.querySelector('#play-btn'),
      retry: document.querySelector('#retry-btn'),
      menuButton: document.querySelector('#menu-btn'),
      hudMenu: document.querySelector('#hud-menu'),

      finalScore: document.querySelector('#final-score'),
      finalBest: document.querySelector('#final-best'),

      hudScore: document.querySelector('#hud-score'),
      hudBest: document.querySelector('#hud-best'),

      devStatus: document.querySelector('#dev-status')
    };

    const saved = store.get('cyber.difficulty', 'EASY');
    this.selectDifficulty(saved);

    document.querySelectorAll('.difficulty-btn').forEach(button => {
      button.addEventListener('click', () => {
        this.selectDifficulty(button.dataset.difficulty);
      });
    });

    this.elements.play.addEventListener('click', () => {
      window.gameScene.setDifficulty(this.selectedDifficulty);
      window.gameScene.startGame('PLAY');
    });

    this.elements.retry.addEventListener('click', () => {
      window.gameScene.startGame('PLAY');
    });

    this.elements.menuButton.addEventListener('click', () => {
      window.gameScene.setSceneState('MENU');
      this.showMenu();
    });

    this.elements.hudMenu.addEventListener('click', () => {
      window.gameScene.setSceneState('MENU');
      this.showMenu();
    });

    document.querySelectorAll('.dev-panel button[data-mode]').forEach(button => {
      button.addEventListener('click', () => {
        window.gameScene.startGame(button.dataset.mode);

        this.elements.devStatus.textContent =
          button.dataset.mode === 'TRAIN'
            ? 'TRAINING'
            : 'AI PLAY';
      });
    });

    document.querySelector('#reset-ai').addEventListener('click', () => {
      window.gameScene.agent.reset();
      store.remove('cyber.model');
      this.elements.devStatus.textContent = 'AI RESET';
    });
  },

  selectDifficulty(name) {
    if (!DIFFICULTIES[name]) return;

    this.selectedDifficulty = name;

    document.querySelectorAll('.difficulty-btn').forEach(button => {
      button.classList.toggle(
        'active',
        button.dataset.difficulty === name
      );
    });

    store.set('cyber.difficulty', name);

    if (window.gameScene) {
      window.gameScene.setDifficulty(name);
    }
  },

  showMenu() {
    this.elements.menu.classList.remove('hidden');
    this.elements.gameOver.classList.add('hidden');
    this.elements.hud.classList.add('hidden');
    this.elements.devPanel.classList.add('hidden');
  },

  showGame() {
    this.elements.menu.classList.add('hidden');
    this.elements.gameOver.classList.add('hidden');
    this.elements.hud.classList.remove('hidden');
    this.elements.devPanel.classList.add('hidden');
  },

  showGameOver(score, best) {
    this.elements.menu.classList.add('hidden');
    this.elements.hud.classList.add('hidden');
    this.elements.gameOver.classList.remove('hidden');

    this.elements.finalScore.textContent = score;
    this.elements.finalBest.textContent = best;
  },

  updateHUD(score, best) {
    this.elements.hudScore.textContent = score;
    this.elements.hudBest.textContent = best;
  }
};

const phaserGame = new Phaser.Game({
  type: Phaser.AUTO,
  parent: 'game',
  width: GAME.width,
  height: GAME.height,
  backgroundColor: '#03050d',

  scale: {
    mode: Phaser.Scale.FIT,
    autoCenter: Phaser.Scale.CENTER_BOTH
  },

  scene: [CyberFlightScene]
});

window.addEventListener('load', () => {
  window.GameUI.init();

  window.gameScene = phaserGame.scene.getScene(
    'CyberFlightScene'
  );

  window.GameUI.showMenu();
});
