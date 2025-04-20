import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import nest_asyncio 
nest_asyncio.apply()


# Load environment variables from .env file
load_dotenv()



# Initialize the bot with intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Load the model and labels
model = tf.keras.models.load_model("keras_model.h5")

def get_class(model, labels_path, image_path):
    # Load the labels
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()

    # Preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to the input size of your model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    return class_index, labels[class_index]  # Return class index and label

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # Bot kendi mesajına cevap vermesin

    if message.attachments:
        for attachment in message.attachments:
            file_name = attachment.filename
            await attachment.save(file_name)  # Save the attachment
            class_index, class_label = get_class(model, "labels.txt", file_name)

            await message.channel.send(f"Bu yemek şu gruba ait: {class_label}")

            extra_info = {
                0: "Yararlı olduğu durumlar: Pizzalar, Ev yapımı, dengeli malzemelerle hazırlanan pizza besleyici olabilir. Riskli olduğu durumlar: Fast food/yağlı pizzalar ise aşırı tüketimde zararlıdır.",
                1: "Yararlı olduğu durumlar: Sushi, Taze, Kaliteli malzemelerle hazırlanmış, düşük cıvalı balık içeren ve aşırı sos eklenmemiş sushi. Riskli olduğu durumlar: Çiğ balığın hijyenik olmadığı, yüksek cıvalı balıkların veya kızartmaların kullanıldığı sushi.",
                2: "Yararlı olduğu durumlar: Tako, Protein ve mineral deposu, düşük kalorili, omega-3 kaynağı. Çiğ ve hijyenik olmayan koşullarda hazırlanmışsa parazit riski var.",
                3: "Yararlı olduğu durumlar: Curry, kargagiller (Corvidae) familyasına ait zeki ve uyum yeteneği yüksek kuşlardır.",
                4: "Yararlı olduğu durumlar: Paela, Taze malzemelerle, sebze ve deniz ürünleri ağırlıklı, esmer pirinçle yapılan paella. Riskli olduğu durumlar: Aşırı yağlı, işlenmiş etli ve beyaz pirinçle yapılan versiyonlar.",
                5: "Yararlı olduğu durumlar: Biryani, Esmer pirinç, bol sebze, baharatlı ve az yağlı biryani. Riskli olduğu durumlar: Beyaz pirinç, bol yağlı, kızartılmış garnitürlü versiyonlar.",
                6: "Yararlı olduğu durumlar: Pekin Duck, Derisiz, az yağlı pişirilmiş ve ölçülü tüketildiğinde iyi bir protein ve mineral kaynağıdır. Riskli olduğu durumlar: Derili, kızartılmış, bol soslu ve aşırı tüketilen versiyonlar sağlık riski oluşturabilir.",
                7: "Yararlı olduğu durumlar: Kebap, Yağsız et, doğru pişirme, sebze ile dengeli tüketim. Riskli olduğu durumlar: Yağlı et, kömürleşmiş, bol ekmek/pilavla yenilen kebap.",
                8: "Yararlı olduğu durumlar: Falafel, Ev yapımı, fırınlanmış, bol sebzeli ve az yağlı falafel. Riskli olduğu durumlar: Kızartılmış, bol soslu ve hazır falafeller."
            }

            info = extra_info.get(class_index, "Sınıflandırma başarısız oldu.")
            await message.channel.send(info)

    await bot.process_commands(message)

# Run the bot
bot.run('BOT TOKEN')
