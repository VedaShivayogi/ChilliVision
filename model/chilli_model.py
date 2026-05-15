"""
Explainable Deep Learning Model for Red Chilli Quality Grading
MobileNetV2 backbone + GradCAM explainability + CV feature extraction
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np, cv2, os

CHILLI_TYPES = {
    0: {"name":"Byadagi Chilli","origin":"Karnataka, India","scoville":"10,000–50,000 SHU","color":"#C1440E","description":"Deep red wrinkled skin, famous for brilliant colour used in masalas. High oleoresin content gives exceptional colouring power without extreme heat."},
    1: {"name":"Kashmiri Chilli","origin":"Kashmir, India","scoville":"1,000–2,000 SHU","color":"#E8300A","description":"Bright scarlet-red, mild heat with fruity undertones. Prized for vivid colour. Key ingredient in Rogan Josh and Kashmiri curries."},
    2: {"name":"Guntur Sannam","origin":"Andhra Pradesh, India","scoville":"35,000–40,000 SHU","color":"#A52A0A","description":"Pungent, sharp heat with deep red colour. One of India's most exported chilli varieties. Backbone of Andhra spice blends."},
    3: {"name":"Kanthari Chilli","origin":"Kerala, India","scoville":"50,000–100,000 SHU","color":"#FF4500","description":"Small bird's eye chilli with explosive heat and intense citrus-like aroma. Used in Kerala cuisine and premium hot sauces."},
    4: {"name":"Wrinkled/Dried Chilli","origin":"Pan-India","scoville":"15,000–30,000 SHU","color":"#8B2500","description":"Sun-dried variety with concentrated smoky flavour. Intensified capsaicin due to moisture loss. Used in pickles, chutneys, and tempering."},
}

QUALITY_GRADES = {
    "Grade A": {"score_range":(85,100),"label":"Premium Quality","color":"#22c55e","description":"Excellent colour uniformity, no defects, optimal moisture content, premium commercial grade."},
    "Grade B": {"score_range":(65,84),"label":"Good Quality","color":"#eab308","description":"Good colour, minor surface blemishes acceptable, suitable for most commercial uses."},
    "Grade C": {"score_range":(40,64),"label":"Average Quality","color":"#f97316","description":"Noticeable discolouration or minor damage, suitable for processing and grinding."},
    "Grade D": {"score_range":(0,39),"label":"Poor Quality","color":"#ef4444","description":"Significant defects, mould, or rot detected. Not recommended for consumption."},
}

class ChilliNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(nn.Dropout(0.3),nn.Linear(1280,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU())
        self.type_head = nn.Linear(256, num_classes)
        self.quality_head = nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())
        self.gradients = None; self.activations = None
        self.features[-1].register_forward_hook(lambda m,i,o: setattr(self,'activations',o.detach()))
        self.features[-1].register_backward_hook(lambda m,gi,go: setattr(self,'gradients',go[0].detach()))
    def forward(self,x):
        f=self.features(x); p=self.avgpool(f); flat=torch.flatten(p,1); s=self.classifier(flat)
        return self.type_head(s), self.quality_head(s)

class GradCAM:
    def __init__(self,model): self.model=model
    def generate(self,tensor,class_idx=None):
        self.model.zero_grad()
        out,_=self.model(tensor)
        if class_idx is None: class_idx=out.argmax(1).item()
        out[0,class_idx].backward()
        g=self.model.gradients; a=self.model.activations
        if g is None or a is None: return None
        cam=F.relu((g.mean([2,3],keepdim=True)*a).sum(1,keepdim=True)).squeeze().cpu().numpy()
        if cam.max()>cam.min(): cam=(cam-cam.min())/(cam.max()-cam.min())
        return cam

def get_transform():
    return transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def preprocess_image(img):
    if isinstance(img,str): img=Image.open(img).convert("RGB")
    else: img=img.convert("RGB")
    return img, get_transform()(img).unsqueeze(0)

def extract_cv_features(pil_img):
    n=np.array(pil_img.resize((224,224)))
    hsv=cv2.cvtColor(n,cv2.COLOR_RGB2HSV); gray=cv2.cvtColor(n,cv2.COLOR_RGB2GRAY)
    r,g,b=n[:,:,0].astype(float),n[:,:,1].astype(float),n[:,:,2].astype(float)
    rd=min(r.mean()/(g.mean()+b.mean()+1e-5)/3.0,1.0)
    sat=hsv[:,:,1].mean()/255.0
    uni=1.0-min(gray.std()/128.0,1.0)
    sharp=min(cv2.Laplacian(gray,cv2.CV_64F).var()/1000.0,1.0)
    _,dm=cv2.threshold(gray,40,255,cv2.THRESH_BINARY_INV)
    defect=max(0.0,1.0-dm.sum()/(255.0*dm.size)*10)
    patches=[gray[i:i+32,j:j+32].std() for i in range(0,192,32) for j in range(0,192,32)]
    tex=1.0-min(np.std(patches)/30.0,1.0)
    return {"red_dominance":round(rd*100,1),"color_saturation":round(sat*100,1),"surface_uniformity":round(uni*100,1),"sharpness":round(sharp*100,1),"defect_free_score":round(defect*100,1),"texture_quality":round(tex*100,1)}

def classify_by_cv(pil_img):
    n=np.array(pil_img.resize((224,224)))
    hsv=cv2.cvtColor(n,cv2.COLOR_RGB2HSV); gray=cv2.cvtColor(n,cv2.COLOR_RGB2GRAY)
    h,s,v=hsv[:,:,0].mean(),hsv[:,:,1].mean(),hsv[:,:,2].mean()
    patches=[gray[i:i+28,j:j+28].std() for i in range(0,196,28) for j in range(0,196,28)]
    tv=np.std(patches)
    sc=np.zeros(5)
    sc[0]=(1-abs(v-130)/130)*0.4+min(tv/40,1)*0.3+(s/255)*0.3
    sc[1]=(s/255)*0.4+(v/255)*0.3+(1-min(tv/40,1))*0.3
    sc[2]=(s/255)*0.5+(1-abs(v-100)/130)*0.5
    sc[3]=(v/255)*0.4+(1-abs(h-5)/30)*0.3+(s/255)*0.3
    sc[4]=(1-v/255)*0.5+min(tv/40,1)*0.5
    e=np.exp(sc-sc.max()); return (e/e.sum()).tolist()

def load_model(model_path=None):
    m=ChilliNet(); 
    if model_path and os.path.exists(model_path):
        m.load_state_dict(torch.load(model_path,map_location="cpu",weights_only=True))
    m.eval(); return m

def cam_to_base64(cam,pil_img):
    import base64,io
    cr=cv2.resize(cam,(224,224))
    hm=cv2.cvtColor(cv2.applyColorMap(np.uint8(255*cr),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    ov=(hm*0.45+np.array(pil_img)*0.55).astype(np.uint8)
    buf=io.BytesIO(); Image.fromarray(ov).save(buf,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def analyse_chilli(pil_img, model, return_gradcam=True):
    pil_r=pil_img.resize((224,224)); _,tensor=preprocess_image(pil_img)
    cv_feat=extract_cv_features(pil_img); cv_probs=classify_by_cv(pil_img)
    with torch.no_grad():
        tl,qr=model(tensor)
    dl_p=F.softmax(tl,1)[0].tolist(); dl_q=float(qr[0][0])*100
    bp=[0.6*cv_probs[i]+0.4*dl_p[i] for i in range(5)]
    bs=sum(bp); bp=[p/bs for p in bp]
    pc=int(np.argmax(bp))
    cv_avg=np.mean([cv_feat["red_dominance"],cv_feat["color_saturation"],cv_feat["defect_free_score"],cv_feat["surface_uniformity"]])
    fq=round(max(10.0,min(0.4*dl_q+0.6*cv_avg,98.0)),1)
    grade="Grade D"
    for g,info in QUALITY_GRADES.items():
        lo,hi=info["score_range"]
        if lo<=fq<=hi: grade=g; break
    gcam_b64=None
    if return_gradcam:
        try:
            tg=tensor.clone().requires_grad_(True)
            cam=GradCAM(model).generate(tg,class_idx=pc)
            if cam is not None: gcam_b64=cam_to_base64(cam,pil_r)
        except: pass
    top3=sorted(enumerate(bp),key=lambda x:x[1],reverse=True)[:3]
    t3=[{"rank":i+1,"type":CHILLI_TYPES[idx]["name"],"confidence":round(p*100,1),"color":CHILLI_TYPES[idx]["color"]} for i,(idx,p) in enumerate(top3)]
    ci=CHILLI_TYPES[pc]; gi=QUALITY_GRADES[grade]
    recs={"Grade A":f"✅ {ci['name']} is premium grade. Ideal for direct export, retail packaging, and premium spice blends.","Grade B":f"👍 {ci['name']} is good quality. Suitable for retail, restaurant supply, and standard spice production.","Grade C":f"⚠️ {ci['name']} is average quality. Best used for industrial grinding, sauces, or chilli powder production.","Grade D":f"❌ {ci['name']} shows poor quality indicators. Recommend rejection or further inspection before use."}
    return {"predicted_type":ci["name"],"type_confidence":round(bp[pc]*100,1),"origin":ci["origin"],"scoville":ci["scoville"],"type_description":ci["description"],"quality_grade":grade,"quality_label":gi["label"],"quality_score":float(fq),"quality_color":gi["color"],"quality_description":gi["description"],"top3_predictions":t3,"cv_features":{k:float(v) for k,v in cv_feat.items()},"gradcam_image":gcam_b64,"is_good":bool(fq>=65),"recommendation":recs.get(grade,"Analysis complete.")}
