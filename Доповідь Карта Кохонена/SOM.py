import numpy as np
import matplotlib.pyplot as plt
import random

# Функція для знаходження (g,h) індексу найближчого нейрона у мережі
def find_BMU(SOM, x):
    distSq = np.sum(np.square(SOM - x), axis=2)  # Обчислюємо квадрат відстані між кожним нейроном і вхідним вектором
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)  # Повертаємо індекс нейрона з найменшою відстанню
    
# Функція для оновлення ваг нейронів при підготовці даних
def update_weights(SOM, train_ex, learn_rate, radius_sq, BMU_coord, step=3):
    g, h = BMU_coord
    # Якщо радіус дуже малий, то оновлюємо тільки BMU (Best Matching Unit - Найкраща відповідна одиниця)
    if radius_sq < 1e-3: #0.001 
        SOM[g,h,:] += learn_rate * (train_ex - SOM[g,h,:]) # Оновлюємо ваги для BMU 
        return SOM 
    # Інакше оновлюємо ваги для всіх нейронів, які знаходяться в радіусі від BMU  
    for i in range(max(0, g-step), min(SOM.shape[0], g+step)): 
        for j in range(max(0, h-step), min(SOM.shape[1], h+step)): 	
            dist_sq = np.square(i - g) + np.square(j - h) # Обчислюємо квадрат відстані між BMU і поточним нейроном
            dist_func = np.exp(-dist_sq / 2 / radius_sq) # Обчислюємо функцію відстані для поточного нейрона 
            SOM[i,j,:] += learn_rate * dist_func * (train_ex - SOM[i,j,:]) # Оновлюємо ваги для поточного нейрона   
    return SOM 

# Основна функція для SOM
def train_SOM(SOM, train_data, learn_rate=0.1, radius_sq=1, 
lr_decay=0.1, radius_decay=0.1, epochs=10):  
    learn_rate_0 = learn_rate # Початкове значення швидкості навчання
    radius_0 = radius_sq # Початкове значення радіусу
    for epoch in range(epochs):  
        random.shuffle(train_data) # Перемішуємо тренувальні дані на початку кожної епохи     
        for train_ex in train_data:   
            g, h = find_BMU(SOM, train_ex) # Знаходимо BMU для кожного прикладу 
            SOM = update_weights(SOM, train_ex, learn_rate, radius_sq, (g,h)) # Оновлюємо ваги нейронів
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay) # Оновлюємо швидкість навчання 
        radius_sq = radius_0 * np.exp(-epoch * radius_decay) # Оновлюємо радіус            
    return SOM

# Розміри мережі SOM
m = 10  
n = 10
# Кількість тренувальних прикладів
n_x = 3000
rand = np.random.RandomState(0)
# Ініціалізуємо тренувальні дані
train_data = rand.randint(0, 255, (n_x, 3))
# Ініціалізуємо SOM випадковими значеннями
SOM = rand.randint(0, 255, (m, n, 3)).astype(float)

# Візуалізація тренувальних даних та мережі SOM
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 3.5), 
                        subplot_kw=dict(xticks=[], yticks=[])) # Вказуємо розміри підграфіків
ax[0].imshow(train_data.reshape(50, 60, 3)) 
ax[0].set_title('Training Data') 
ax[1].imshow(SOM.astype(int)) 
ax[1].set_title('Randomly Initialized SOM Grid')

  

# Візуалізація кожні 5 епох, щоб отримати короткий огляд прогресу SOM
fig, ax = plt.subplots(
    nrows=1, ncols=4, figsize=(15, 3.5), 
    subplot_kw=dict(xticks=[], yticks=[])) # Вказуємо розміри підграфіків
total_epochs = 0 # Ініціалізуємо загальну кількість епох
for epochs, i in zip([1, 4, 5, 10], range(0,4)): # Вказуємо кількість епох для кожного підграфіка
    total_epochs += epochs # Обчислюємо загальну к-сть епох
    SOM = train_SOM(SOM, train_data, epochs=epochs) # Навчаємо мережу SOM на вказану к-сть епох
    ax[i].imshow(SOM.astype(int))  # Відображаємо стан мережі SOM
    ax[i].title.set_text('Epochs = ' + str(total_epochs))  # Встановлюємо заголовок для підграфіка, щоб відображати к-сть пройдених епох



# Візуалізація навченої мережі SOM з різними швидкостями навчання та радіусами
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), 
                       subplot_kw=dict(xticks=[], yticks=[])) # Вказуємо розміри підграфіків

for learn_rate, i in zip([0.001, 0.5, 0.99], range(3)): # Вказуємо швидкості навчання для кожного рядка  
    for radius_sq, j in zip([0.01, 1, 10], range(3)): # Вказуємо радіуси для кожного стовпчика 
        rand = np.random.RandomState(0) # Ініціалізуємо генератор випадкових чисел 
        SOM = rand.randint(0, 255, (m, n, 3)).astype(float) # Ініціалізуємо мережу SOM випадковими значеннями     
        SOM = train_SOM(SOM, train_data, epochs=5, 
                        learn_rate=learn_rate, 
                        radius_sq=radius_sq) # Навчаємо мережу SOM на 5 епох з вказаною швидкістю навчання та радіусом
        ax[i][j].imshow(SOM.astype(int)) # Відображаємо стан мережі SOM
        ax[i][j].set_title(r'$\eta$ = {:.3f}, $\sigma^2$ = {}'.format(learn_rate, radius_sq)) # Встановлюємо заголовок для підграфіка


plt.show()
