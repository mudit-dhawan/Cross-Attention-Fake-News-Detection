import torch
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from transformers import BertModel
import math


# Create the Bert custom class 
class TextEncoder(nn.Module):

    def __init__(self, fine_tune_module=False):

        super(TextEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(
                    'bert-base-uncased',
#                     output_attentions = True, 
                    return_dict=True)


        self.fine_tune()
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        ## odict_keys(['last_hidden_state', 'pooler_output', 'attentions'])
        ## last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 

        return out    
    
    def fine_tune(self):
        """
        keep the weights fixed or not  
        """
        for p in self.bert.parameters():
            p.requires_grad = False
           
        for c in list(self.bert.children())[-3:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module
            

            
class VisionEncoder(nn.Module):
    """Visual Feature extraction
    """
    def __init__(self, encoded_fc_dim=512, encoded_image_size=14, fine_tune_module=False):
        """

        """
        super(VisionEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module
        self.encoded_image_size = encoded_image_size

        
        # Instantiate Resnet model
        resnet = models.resnet18(pretrained=True)
        
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool_cnn = nn.AdaptiveAvgPool2d((self.encoded_image_size, self.encoded_image_size))

        ## Create 1D dimensional Embedding 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))


        self.fine_tune()
        

        
    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)

        ## keeping CNN output for attention maps
        out_cnn = self.adaptive_pool_cnn(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out_cnn = out_cnn.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 512)

        ## 1D vector representation 
        out = self.adaptive_pool(out)

        return out, out_cnn
    
    def fine_tune(self):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module


                
class CrossAttentionModule(nn.Module):
    
    def __init__(self, text_enc_dim, img_enc_dim, comb_enc_dim, att_dim):
        super(CrossAttentionModule, self).__init__()
        
        ## Weights to convert embeddings into keys and queries
        self.img_key = nn.Linear(img_enc_dim, att_dim)
        self.text_key = nn.Linear(text_enc_dim, att_dim)
        self.comb_query = nn.Linear(comb_enc_dim, att_dim)

    ## Function to create the queries and keys 
    def create_k_q(self, enc_img, enc_text, combined_enc):
        text_k = self.text_key(enc_text)
        img_k = self.img_key(enc_img)
        comb_q = self.comb_query(combined_enc)

        return text_k, img_k, comb_q
    


    def forward(self, query, key, value, mask=None, dropout=None):

        d_k = query.size(-1)
        query = query.unsqueeze(1)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = nn.functional.softmax(scores, dim = -1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        new_p_attn = p_attn.squeeze(1).unsqueeze(2)

        return torch.matmul(p_attn, value), new_p_attn
#         return (new_p_attn*value), new_p_attn


class PostAttention(nn.Module):
    def __init__(self, text_emb_size, text_enc_dim, hidden_size, img_enc_cnn, vis_emb_size, vis_enc_dim):
        super(PostAttention, self).__init__()

#         ## Accumulating Post-Attention text features 
#         self.text_encoder = nn.LSTM(text_emb_size, hidden_size, num_layers=1, bidirectional=False,
#                                batch_first=True)

        ## Latent space for text features 
        self.text_enc_fc = torch.nn.Linear(text_emb_size, text_enc_dim)

#         ## Accumulating Post-Attention visual features  
#         self.img_enc_cnn = img_enc_cnn
#         self.adaptive_pool_vis = nn.AdaptiveAvgPool2d((1, 1))

        ## Latent space for Visual features 
        self.vis_enc_fc = nn.Linear(vis_emb_size, vis_enc_dim)
    
    def forward(self, x_text, x_vis):
        batch_size = x_vis.size(0)
        
        ## Text features 
#         print("X text before lstm", x_text.shape)
#         _, (hidden, _not) = self.text_encoder(x_text)
        
#         hidden = hidden.squeeze()
        x_text = x_text.view(batch_size, -1)
        x_text = self.text_enc_fc(x_text)

        ## Visual Features 
#         img_feature_size = x_vis.size(-1)
        
        
#         x_vis = x_vis.contiguous().view(batch_size, img_feature_size, self.img_enc_cnn, self.img_enc_cnn)
# #         print("x vis before pool", x_vis.shape)
#         x_vis = self.adaptive_pool_vis(x_vis)
#         print("x vis after pool", x_vis.shape)
        
        x_vis = x_vis.view(batch_size, -1)
#         print("x vis after", x_vis.shape)
        x_vis = self.vis_enc_fc(x_vis)

        return x_text, x_vis
                
class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        model_params
        
    ):
        super(LanguageAndVisionConcat, self).__init__()
        
        ## Text encoder 
        self.text_encoder = TextEncoder(model_params['fine_tune_text_module'])
        
        ## Visual Encoder 
        self.vision_encoder = VisionEncoder(model_params['img_enc_fc_dim'], model_params['img_enc_cnn'], 
                                            model_params['fine_tune_vis_module'])
        
        ## Cross-Attention Module 
        self.attention_module = CrossAttentionModule(model_params['text_enc_dim'], model_params['img_enc_dim'], 
                                                     model_params['img_enc_fc_dim']+model_params['text_enc_dim'], 
                                                     model_params['att_dim'])
        
        ## Post-Attention module 
        self.post_attention_module = PostAttention(model_params['text_emb_size'], model_params['text_latent_dim'], 
                                                   model_params['hidden_size'], model_params['img_enc_cnn'], 
                                                   model_params['vis_emb_size'],model_params['vis_enc_dim']) 
        
        self.fusion = torch.nn.Linear(
            in_features=(model_params['text_latent_dim'] + model_params['vis_enc_dim']), 
            out_features=model_params['fusion_output_size']
        )
        self.fc = torch.nn.Linear(
            in_features=model_params['fusion_output_size'], 
            out_features=model_params['num_classes']
        )
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])
        
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((256, 512))
        

    def prep_attention(self, enc_img_cnn, enc_img_fc, enc_text):

        ## Image Modality 
        ## Image embedding dimensions 
        img_feature_size = enc_img_cnn.size(-1)
        batch_size = enc_img_cnn.size(0)
        
        ## Change fc embedding from (bs, enc dim, 1, 1) -> (bs, enc_dim)
        enc_img_fc = enc_img_fc.view(batch_size, -1)
        
        # Flatten image (img_feature_size = number feature maps in cnn)
        att_ready_img = enc_img_cnn.view(batch_size, -1, img_feature_size)  # (batch_size, num_pixels, encoder_dim)
#         print("Att ready image size", att_ready_img.shape)
        
        ## Text Modality 
        ## taking [cls] token embedding
        enc_text_cls = enc_text.last_hidden_state[:, 0, :]

        ## taking other tokens embedding
        att_ready_text = enc_text.last_hidden_state[:, 1:, :]
#         print("Att ready text size", att_ready_text.shape)
#         att_ready_text = self.adaptive_pool(att_ready_text.unsqueeze(1)).squeeze(1)
        
                                            
#         print("enc_text_cls", enc_text_cls.shape, "enc_img_fc", enc_img_fc.shape)
        ## concatenating Image and text 
        att_ready_comb = torch.cat(
            [enc_text_cls, enc_img_fc], dim=1
        )

        return att_ready_img, att_ready_text, att_ready_comb

    def forward(self, text, image, label=None):

        ## Pass the text input to Bert encoder 
        text_features = self.text_encoder(text[0], text[1])
#         print("Text Features after BERT: ", text_features) 

        ## Pass the image input 
        image_features, image_features_cnn = self.vision_encoder(image)
#         print("Image featrues after fc encoding :", image_features.shape, "Image featrues after cnn encoding :", image_features_cnn.shape)

        ## Create attention ready features 
        att_ready_img, att_ready_text, att_ready_comb = self.prep_attention(image_features_cnn, image_features, text_features)
#         print("Attention ready: image -", att_ready_img.shape," | Text -", att_ready_text.shape, " | Combined -", att_ready_comb.shape)
        
        ## Create queries and keys
        text_key, img_key, comb_query = self.attention_module.create_k_q(att_ready_img, att_ready_text, att_ready_comb)
#         print("Text key :", text_key.shape, " | Image key :", img_key.shape, " | Combined query :", comb_query.shape)
        
        ## Attention map on Images 
        attn_image_features, att_mask_imgs = self.attention_module(comb_query, img_key, value=att_ready_img)
#         print("Image Att Features :", attn_image_features.shape, " | Att Map :", att_mask_imgs.shape)
        
        ## Attention map on Text 
        attn_text_features, att_mask_text = self.attention_module(comb_query, text_key, value=att_ready_text)
#         print("Text Features after BERT: ", text_feature)
#         print("Text Att Features :", attn_text_features.shape, " | Att Map :", att_mask_text.shape)
        
        ## Final with attention text and visual feature 
        text_final_emb, vis_final_emb = self.post_attention_module(attn_text_features, attn_image_features)
#         print("text final embedding :", text_final_emb.shape," | final Visual embedding :", vis_final_emb.shape)     

        ## concatenating Image and text 
        combined_features = torch.cat(
            [text_final_emb, vis_final_emb], dim=1
        )        
#         print("Combined Features :", combined_features.shape)
        

        fused = self.dropout(
            torch.nn.functional.relu(
            self.fusion(combined_features)
            )
        )
#         print("Fused Features :", fused.shape)
        
        logits = self.fc(fused)

        return logits, att_mask_imgs, att_mask_text           