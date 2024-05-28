import fullbody_movement_tracker.movenet as movenet

modelGetter = movenet.Detector(isTflite=True)

test = modelGetter.loadModel("movenet_lightning_f16")
test = modelGetter.doImageInference('test.jpg')

print(test)