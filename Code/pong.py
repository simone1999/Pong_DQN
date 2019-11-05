import pygame
import numpy as np
import random

class Pong:
    def __init__(self, numGames):
        self.numGames = numGames
        self.gameHeight = 128
        self.gameWidth = 128
        self.ballMomentums = np.zeros(shape=(self.numGames, 2), dtype=np.float)
        self.ballPositions = np.zeros(shape=(self.numGames, 2), dtype=np.float)
        self.paddlePositions = np.zeros(shape=(self.numGames, 2), dtype=np.float)

        self.paddleClearance = 16
        self.paddleWidth = 5
        self.paddleHeight = 16
        self.paddleSpeed = 2
        self.ballSize = 2
        self.bounceRandomness = 0.1
        self.ballSpeedup = 1.0001

        self.resetGames(np.ones(shape=self.numGames, dtype=bool))

        self.scaleFactor = 4
        window_size = (self.gameWidth*self.scaleFactor, self.gameHeight*self.scaleFactor)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Pong")

    def step(self, actionsLeft, actionsRight):
        self.ballMomentums *= self.ballSpeedup

        self.paddlePositions[actionsLeft == 1, 0] = self.paddlePositions[actionsLeft == 1, 0] + self.paddleSpeed
        self.paddlePositions[actionsLeft == 2, 0] = self.paddlePositions[actionsLeft == 2, 0] - self.paddleSpeed

        self.paddlePositions[actionsRight == 1, 1] = self.paddlePositions[actionsRight == 1, 1] + self.paddleSpeed
        self.paddlePositions[actionsRight == 2, 1] = self.paddlePositions[actionsRight == 2, 1] - self.paddleSpeed

        maxPaddleCenterHeight = self.gameHeight - (self.paddleHeight // 2)
        minPaddleCenterHeight = self.paddleHeight // 2

        self.paddlePositions[self.paddlePositions > maxPaddleCenterHeight] = maxPaddleCenterHeight
        self.paddlePositions[self.paddlePositions < minPaddleCenterHeight] = minPaddleCenterHeight

        beforeLeftPaddleMask = self.ballPositions[:, 0] > self.paddleClearance + self.ballSize
        beforeRightPaddleMask = self.ballPositions[:, 0] < self.gameWidth - (self.paddleClearance + self.ballSize)

        self.ballPositions += self.ballMomentums

        self.ballMomentums[self.ballPositions[:, 1] >= self.gameHeight, 1] *= -1
        self.ballMomentums[self.ballPositions[:, 1] <= 0, 1] *= -1

        nowAfterLeftPaddleMask = self.ballPositions[:, 0] <= self.paddleClearance + self.ballSize
        nowAfterRightPaddleMask = self.ballPositions[:, 0] >= self.gameWidth - (self.paddleClearance + self.ballSize)

        leftPaddleLineTouches = np.logical_and(beforeLeftPaddleMask, nowAfterLeftPaddleMask)
        rightPaddleLineTouches = np.logical_and(beforeRightPaddleMask, nowAfterRightPaddleMask)

        infrontOfLeftPaddle = np.logical_and(self.ballPositions[:, 1] > self.paddlePositions[:, 0] - self.paddleHeight/2, self.ballPositions[:, 1] < self.paddlePositions[:, 0] + self.paddleHeight/2)
        infrontOfRightPaddle = np.logical_and(self.ballPositions[:, 1] > self.paddlePositions[:, 1] - self.paddleHeight/2, self.ballPositions[:, 1] < self.paddlePositions[:, 1] + self.paddleHeight/2)

        leftPaddleTouches = np.logical_and(leftPaddleLineTouches, infrontOfLeftPaddle)
        rightPaddleTouches = np.logical_and(rightPaddleLineTouches, infrontOfRightPaddle)

        paddleTouches = np.logical_or(leftPaddleTouches, rightPaddleTouches)

        if np.any(paddleTouches):
            oldMomentum = np.sqrt(np.add(np.square(self.ballMomentums[paddleTouches, 0]), np.square(self.ballMomentums[paddleTouches, 1])))

            leftPaddleHitpoint = (self.ballPositions[leftPaddleTouches, 1] - self.paddlePositions[leftPaddleTouches, 0]) / (self.paddleHeight / 2)
            rightPaddleHitpoint = (self.ballPositions[rightPaddleTouches, 1] - self.paddlePositions[rightPaddleTouches, 1]) / (self.paddleHeight / 2)
            self.ballMomentums[leftPaddleTouches, 1] += leftPaddleHitpoint * 1
            self.ballMomentums[rightPaddleTouches, 1] += rightPaddleHitpoint * 1

            self.ballMomentums[paddleTouches, 0] *= -1
            self.ballMomentums[paddleTouches, 1] += (2 * random.random() - 1) * self.bounceRandomness

            newMomentum = np.sqrt(np.add(np.square(self.ballMomentums[paddleTouches, 0]), np.square(self.ballMomentums[paddleTouches, 1])))
            momentumScaler = oldMomentum / newMomentum
            self.ballMomentums[paddleTouches, 0] *= momentumScaler
            self.ballMomentums[paddleTouches, 0] *= momentumScaler


        leftBallOut = self.ballPositions[:, 0] < 0
        rightBallOut = self.ballPositions[:, 0] > self.gameWidth

        rewards = np.zeros(shape=(self.numGames, 2))
        dones = np.zeros(shape=(self.numGames), dtype=np.bool)

        rewards[leftBallOut, 0] += -1
        rewards[leftBallOut, 1] += 1

        rewards[rightBallOut, 0] += 1
        rewards[rightBallOut, 1] += -1

        dones[leftBallOut] = True
        dones[rightBallOut] = True

        self.resetGames(dones)

        states = np.zeros(shape=(self.numGames, 6))
        states[:, 0] = (self.ballPositions[:, 0] / self.gameWidth - 0.5) / 2
        states[:, 1] = (self.ballPositions[:, 1] / self.gameHeight - 0.5) / 2
        states[:, 2] = self.ballMomentums[:, 0] / 5
        states[:, 3] = self.ballMomentums[:, 1] / 5
        states[:, 4] = (self.paddlePositions[:, 0] / self.gameHeight - 0.5) / 2
        states[:, 5] = (self.paddlePositions[:, 1] / self.gameHeight - 0.5) / 2

        return states, rewards, dones


    def resetGames(self, mask):
        numResets = np.sum(mask)
        self.ballMomentums[mask, 0] = 1.5 + np.random.rand(numResets)
        self.ballMomentums[mask, 1] = 0
        self.ballPositions[mask, 0] = self.gameWidth // 2
        self.ballPositions[mask, 1] = self.gameHeight // 2
        self.paddlePositions[mask] = self.gameHeight // 2


    def render(self):
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:
                exit(88)
        self.screen.fill((150, 150, 150))

        pygame.draw.rect(self.screen, (255, 0, 0), [(self.paddleClearance-5)*self.scaleFactor, (self.paddlePositions[0][0] - self.paddleHeight//2)*self.scaleFactor, 5*self.scaleFactor, self.paddleHeight*self.scaleFactor], 0)
        pygame.draw.rect(self.screen, (0, 0, 255), [(self.gameWidth-self.paddleClearance)*self.scaleFactor, (self.paddlePositions[0][1] - self.paddleHeight//2)*self.scaleFactor, 5*self.scaleFactor, self.paddleHeight*self.scaleFactor], 0)

        pygame.draw.circle(self.screen, (255, 255, 0), [int(self.ballPositions[0][0]*self.scaleFactor), int(self.ballPositions[0][1]*self.scaleFactor)], self.ballSize*self.scaleFactor, 0)

        pygame.display.flip()

        #self.screen





