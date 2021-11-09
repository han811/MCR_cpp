import pygame
from pygame.locals import *
from Box2D import *
from Box2D.b2 import *

SCREEN_WD = 640
SCREEN_HT = 480
TARGET_FPS = 60
PPM = 20.0

screen = pygame.display.set_mode((SCREEN_WD, SCREEN_HT), 0, 32)
pygame.display.set_caption("PyBox2D_Example")
clock = pygame.time.Clock()

world = b2World(gravity = (0, 0), doSleep = True)

ground1BodyDef = b2BodyDef()
ground1BodyDef.position.Set(0, 0)
ground1Body = world.CreateBody(ground1BodyDef)
ground1Shape = b2PolygonShape()
ground1Shape.SetAsBox(50, 1)
ground1Body.CreateFixture(shape = ground1Shape)

box1BodyDef = b2BodyDef()
box1BodyDef.type = b2_dynamicBody
box1BodyDef.position.Set(20, 0.5)
box1BodyDef.angle = 15
box1Body = world.CreateBody(box1BodyDef)
box1Shape = b2PolygonShape()
box1Shape.SetAsBox(2, 1)
box1FixtureDef = b2FixtureDef()
box1FixtureDef.shape = box1Shape
box1FixtureDef.density = 1
box1FixtureDef.friction = 0.3
box1FixtureDef.filter.categoryBits = 0x0002
box1FixtureDef.filter.maskBits = 0x0002 | 0x0001
box1Body.CreateFixture(box1FixtureDef)

circle1BodyDef = b2BodyDef()
circle1BodyDef.type = b2_dynamicBody
circle1BodyDef.position.Set(20, 0.5)
circle1Body = world.CreateBody(circle1BodyDef)
circle1Shape = b2CircleShape()
circle1Shape.radius = 0.5
circle1FixtureDef = b2FixtureDef()
circle1FixtureDef.shape = circle1Shape
circle1FixtureDef.density = 1
circle1FixtureDef.friction = 0.3
circle1FixtureDef.filter.categoryBits = 0x0004
circle1FixtureDef.filter.maskBits = 0x0004 | 0x0001
circle1Body.CreateFixture(circle1FixtureDef)

class MyContactListener(b2ContactListener):
    def BeginContact(self, contact):
        print("BeginContact")
    def EndContact(self, contact):
        print("EndContact")

contactListener = MyContactListener()
world.contactListener = contactListener

timeStep = 1.0 / 60
velIters = 10
posIters = 10

colors = {
    staticBody  : (255,255,255,255),
    dynamicBody : (127,127,127,255),
}

def my_draw_polygon(polygon, body, fixture):
    vertices=[(body.transform*v)*PPM for v in polygon.vertices]
    vertices=[(v[0], SCREEN_HT-v[1]) for v in vertices]
    pygame.draw.polygon(screen, colors[body.type], vertices)
polygonShape.draw=my_draw_polygon

def my_draw_circle(circle, body, fixture):
    position=body.transform*circle.pos*PPM
    position=(position[0], SCREEN_HT-position[1])
    pygame.draw.circle(screen, colors[body.type], [int(x) for x in position], int(circle.radius*PPM))
circleShape.draw=my_draw_circle

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
            continue
        if event.type == KEYDOWN and event.key == K_ESCAPE:
            running = False
            continue

    screen.fill((0, 0, 0, 0))

    for body in world.bodies:
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture)
                     
    world.Step(timeStep, velIters, posIters)
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
print("done")
