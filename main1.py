import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchvision import transforms

# 評価用のデータセットクラス
class VQAEvalDataset(Dataset):
    def __init__(self, df_path, image_dir, tokenizer, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question_text = self.df["question"][idx]
        question = self.tokenizer(question_text, return_tensors="pt", padding='max_length', truncation=True, max_length=20)
        return image, question

    def __len__(self):
        return len(self.df)

# 予測結果を取得して保存する関数
def save_predictions(model, data_loader, idx2answer, output_file='submission.npy'):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for images, questions in data_loader:
            images = images.cuda()
            questions = {key: val.squeeze(1).cuda() for key, val in questions.items()}
            outputs = model(images, questions)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    np.save(output_file, all_predictions)

# メイン処理
if __name__ == "__main__":
    # モデルの準備
    vocab_size = len(train_dataset.answer2idx)
    model = VQAModel(vocab_size).cuda()
    model.load_state_dict(torch.load('path/to/your/model.pth'))  # モデルの重みをロード
    model.eval()

    # データセットとデータローダの準備
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    eval_dataset = VQAEvalDataset('eval.json', 'eval_images', tokenizer, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # 予測結果の保存
    save_predictions(model, eval_loader, train_dataset.idx2answer)
    print("Predictions saved to submission.npy")


