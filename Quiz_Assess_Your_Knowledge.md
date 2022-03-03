# Quiz : Assess Your Knowledge

## Question 1
By using albumentation library for image-mask problem, the augmentation parameter will be same of image and its mask. 

For example, if an image rotates at 50 degree its mask will also get rotate at 50 degree.
* <b> True </b> 
* False

## Question 2
Which train_fn() is correct ?
* <b>(a)-</b>
```python
def train_fn(model, trainloader, optimizer):
    
    train_loss = 0.0
    model.train()
    
    for data in tqdm(trainloader):
        
        images, masks = data
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        out_key, loss = model(images, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    return train_loss/len(trainloader)
```
* (b)-
```python
def train_fn(model, trainloader, optimizer):
    
    train_loss = 0.0
    model.train()
    
    for data in tqdm(trainloader):
        
        images, masks = data
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        out_key, loss = model(images, masks)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    return train_loss/len(trainloader)
```

* (c)-
```python
def train_fn(model, trainloader, optimizer):
    
    train_loss = 0.0
    model.train()
    
    for data in tqdm(trainloader):
        
        images, masks = data
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        out_key, loss = model(images, masks)
       
        train_loss += loss.item()
        
    return train_loss/len(trainloader)
 
```
* None of these

## Question 3
In this project, which loss was used for segmentation problem ? 
* <b> nn.BCEWithLogitsLoss() </b>
* <b> DiceLoss(mode = 'binray') </b>
* nn.MSELoss()
* None of these

## Question 4
Mostly for segmentation problem, input image size and mask image size should be same.
* <b> True </b>
* False

## Question 5
In above code, why image.unsqueeze(0) is applied when sending as an input image in the model 
```python
idx = 20

model.load_state_dict(torch.load('/content/best_model.pt'))
image, mask = validset[idx]

#### HERE ####
logits_mask = model(image.to(CFG.DEVICE).unsqueeze(0))
##############
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5)*1.0

```
* It is used to add an extra dimension for channel as model takes (channel, height, weight, batch)
* <b> It is used to add an extra dimension for batch as model takes (batch, channel, height, width)</b>






