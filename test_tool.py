import torch
import torch.nn as nn
import cv2
import numpy as np 

print("Welcome to the test tool!")
print("Please make sure that mnist_nn.pt exists in the current working directory.")
print()
print("CONTROLS:")
print("<esc> to exit.")
print("<backspace> to clear canvas.")
print()
print("Press <enter> to continue...")
input()

model = nn.Sequential(
    nn.Linear(784, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
    nn.Softmax(dim = 0)
)
model.load_state_dict(torch.load("mnist_nn.pt"))

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)        


img = np.zeros((28,28,3), np.uint8)
cv2.namedWindow('Draw a Digit Here!')
cv2.setMouseCallback('Draw a Digit Here!',line_drawing)

print()
while(1):
    inpt = img[:, :, 0].squeeze()
    logits = model(torch.tensor(inpt, dtype=torch.float32).view(784))
    val, idx = torch.topk(logits, 1)
    print(f"Model Prediction: {idx.item()}\r", end="")

    cv2.imshow(f'Draw a Digit Here!',img)
    k = cv2.waitKey(1)
    if k & 0xFF == 27:
        break
    elif k & 0xFF == 8:
        img = np.zeros((28,28,3), np.uint8)
cv2.destroyAllWindows()