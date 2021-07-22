'''
Author: Ambareesh Ravi
Date: July 20, 2021
Title: prediction.py
Project: For InvisionAI recuritment
Description:
    Module to run evaluation and prediction on the trained PixelClassification CNN model
'''

# Module imports
from train import *

# Class for prediction/inference, evaluation, visualization of results
class PixelClassifier_Tester(Rooftop_Dataset):
    def __init__(
        self,
        model = None,
        model_path = "models/model_v5.pth",
        loss_params = {
         "loss_type": "bce",
         "reduction": "sum"
        },
        max_pool = True,
        useGPU = True,
        show_status = True
    ):
        '''
        Tester module for PixelClassifier CNN model
        
        Args:
            model - [Optional] The trained <torch.nn.Module> model
            model_path - path of the trained model as <str>
            loss_params - <dict> containing params for the loss function
            max_pool - <bool> applies max pool layer at the end
            useGPU - <bool> to utilize GPU capabilities
            show_status - <bool> display status
            
        Returns:
            -
            
        Exception:
            -
        '''
        Rooftop_Dataset.__init__(self, isTrain = False)
        self.model = model
        self.model_path = model_path
        if not self.model and self.model_path:
            self.load_model()
        
        self.loss_params = loss_params
        self.loss_criterion = get_loss(self.loss_params["loss_type"], self.loss_params["reduction"])
        self.useGPU = useGPU
        self.show_status = show_status
        
        self.device = torch.device("cpu")
        if self.useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.show_status: INFO("Using GPU for inferencing and testing")
        
        self.max_pool = max_pool
        if self.max_pool: self.maxPool = nn.MaxPool2d((max_pool,max_pool), stride = 1, padding = max_pool//2)
        
    def load_model(self):
        '''
        Loads model into memory
        
        Args:
            -
            
        Returns:
            -
            
        Exception:
            -
        '''
        self.model = torch.load(self.model_path)
        self.model.eval()
        
    def predict(self, inputs):
        '''
        Runs prediction for given inputs
        
        Args:
            inputs - <torch.Tensor>
            
        Returns:
            outputs as <torch.Tensor>
            
        Exception:
            -
        '''
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device))
            if self.max_pool: outputs = self.maxPool(outputs)
        return outputs.cpu()
    
    def evaluate(self, test_loader):
        '''
        Evaluates the test dataset
        
        Args:
            test_loader - data loader for the test set as <torch.utils.data.Dataloader>
            
        Returns:
            average loss as <float>
            
        Exception:
            -
        '''
        losses = list()
        for batch_idx, (test_data, test_labels) in tqdm(enumerate(test_loader)):
            predicted_masks = self.predict(test_data.to(self.device))
            loss = self.loss_criterion(predicted_masks, test_labels)
            losses.append(loss.item())
        
        average_loss = np.mean(losses)
        INFO("TEST %s LOSS is %0.4f"%(self.loss_params["loss_type"].capitalize(), average_loss))
        return average_loss
    
    def process_image(self, image):
        '''
        Preprocess the image for prediction
        
        Args:
            image - input as <str>/ <np.array> 
            
        Returns:
            transformed image as <torch.Tensor>
            
        Exception:
            -
        '''
        if isinstance(image, str): image = read_image(image)
        elif isinstance(image, np.array): image = Image.fromarray(image)
        return self.transform_image(image).unsqueeze(dim = 0)
    
    def predict_image(self, image, return_type = "PIL.Image"):
        '''
        Predicts the image for prediction
        
        Args:
            image - input as <str>/ <np.array> 
            return_type - "np.array"/ "torch.Tensor"/ "PIL.Image"
            
        Returns:
            output
            
        Exception:
            -
        '''
        # input - str, image, np.array, torch.Tensor
        output = self.predict(self.process_image(image))
        output = output.cpu().detach()
        output = output.permute(0,2,3,1).squeeze(dim = 0)
        
        if return_type == "np.array":
            return output.numpy()
        elif return_type == "PIL.Image":
            return Image.fromarray(output)
        else: # torch.Tensor
            return output
        
    def inference_large(self, image_path, patch_size = (500, 500), output_path = "output"):
        '''
        Runs inference on the large image
        
        Args:
            image_path - path to the large input image as <str>
            patch_size - <tuple> containting the patch resolution for inferencing
            output_path - path to save the output as <str>
            
        Returns:
            <PIL.Image>
            
        Exception:
            -
        '''
        img = read_image(image_path).convert("RGB")
        img_array = np.array(img)
        
        image_transform = transforms.Compose([transforms.ToTensor()])
        
        row_tensors = list()
        
        for xidx in range(0, img_array.shape[0], patch_size[0]):
        
            input_patches = list()
            for yidx in range(0, img_array.shape[1], patch_size[1]):
                crop = img_array[xidx:(xidx + patch_size[0]), yidx:(yidx + patch_size[1])]
                input_patches.append(image_transform(Image.fromarray(crop)))
            
            row_outputs = self.predict(torch.stack(input_patches)).detach()
            row_tensors.append(torch.hstack([r for r in row_outputs.squeeze()])) # (image_size / patch_size)  x patch_size x patch_size
        
        output_mask = (np.round(torch.vstack(row_tensors).numpy()) * 255).astype(np.uint8)
        try:
            save_image(output_mask, output_path)
        except:
            pass
        
        return Image.fromarray(output_mask)
        
if __name__=='__main__':
    # Parse input arguments from the user for traing the model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default = "models/model_v5.pth", help="Path of the model to be tested")
    parser.add_argument("--batch_size", type=int, default = 32, help="Batch Size for training")
    parser.add_argument("--evaluate", type=bool, default = False, help="Should the model be evaluated for loss")
    parser.add_argument("--source_image", type=str, default = "source_data/image.tif", help="Source image to run the inference on")
    parser.add_argument("--output_path", type=str, default = "output.png", help="Path to save the final output")
    args = parser.parse_args()
    
    # Load the test dataset
    test_dataset = Rooftop_Dataset(isTrain = False)
    # Get the test_loader
    test_loader = get_data_loader(test_dataset, isTrain = False, batch_size = args.batch_size)
    
    # Create instance of the tester
    tester = PixelClassifier_Tester(model_path = args.model_path)
    
    if args.evaluate:
        # Evaluate the model
        loss = tester.evaluate(test_loader)
    # Run the model on the source input image and save the output
    out = tester.inference_large(image_path = args.source_image, output_path = args.output_path)