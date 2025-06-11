

<p align="center">
  <a href="https://github.com">
    <img src="https://github.com/user-attachments/assets/2006d10c-e253-441a-b23f-69ab64014b55" alt="Logo" width="600"/>
  </a>
</p>

# 📘 Оцінка впливу методів навчання на якість анімації симульованих персонажів


> 
Реалізація методу курування за зразком: [SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning](https://theorangeduck.com/media/uploads/other_stuff/SuperTrack.pdf)

---

## 👤 Автор

- **ПІБ**: Лещенко Руслан Олександрович
- **Група**: ФЕІ-43
- **Керівник**: Фургала Юрій Михайлович, к.-ф. м. н., доц.
- **Дата виконання**: [5.06.25]

---

## Загальна інформація

- **Тип проєкту**: Система анімації персонажів
- **Мова програмування**: python
- **Фреймворки / Бібліотеки**: PyBullet, OpenAI Gym, Pytorch

---

## Набір даних
У цій роботі було адаптовано загальновідомий набір даних [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset). Всі дані захоплення руху представлені у форматі BVH. 


## Реалізація методу включає:

- Власне GYM середовище, розроблене на основі PyBullet  
- URDF модель що відповідає ірерархії скелету LAFAN1 
- Генерацію кінематичного та симульованого руху 
- Бібліотку необхідних функцій перетворень та математичних операцій
- Оптимізоване навчання з використанням паралельного обчислення

---

## Опис основних файлів

|  Файл     | Призначення |
|----------------|-------------|
| `supertrack2.py`      | містить реалізацію методу навчання |
| `SuperTrack\supertrack\resources\character_k.py`  `SuperTrack\supertrack\resources\character.py`| містить клас симульованого та кінематичного персонажа  |
| `SuperTrack\supertrack\envs\supertrack_env.py` | містить клас тренувального середовища |
| `11s3_I.urdf` | URDF модель персонажа |
| `kinematicBVH_TEST_PD.py` `kinematicBVH2.py` | реалізація симульваного та кінематичного контролю |
| `my_lib_t.py` | власна бібліотека |

---

## Як запустити проєкт 

### 1. Клонування репозиторію

```bash
git clone https://github.com/ruslanLeshh/my_supertrack.git
```

### 2. Встановлення бібліотек та середовища

- `python -m venv env` створення віртуального середовища
- `env/Scripts/activate` активація віртуального середовища
- `pip install -r requirements.txt` завантаження потрібних бібліотек
- `pip install -e ./SuperTrack` завантаження тренувального середовища

### 3.  Тестування
(запуск файлу`supertrack2.py` демонтструє режим тестування за замовчуванням)

щоб протестувати роботу агента потрібно:


    env = gym.make("SuperTrack-v0", render_mode="human") #ініціалізувати тренувальне середовище та агента
    agent = SUPERTRACK(env)

    agent.world.eval() # перевести мережі у тестовий режим
    agent.policy.eval()

    agent.gater_data(100) # викликати метод для збору нових епізодів з заданою кількістю фреймів

для тренування необхідно зберегти отримані фрейми використовуючи метод `agent.save_m('назва_буффера.pt')`

### 4. Тренування

для тренування агента потрібно:

    env = gym.make("SuperTrack-v0") #ініціалізувати тренувальне середовище та агента
    agent = SUPERTRACK(env)

    agent.load_m('supertrack_60test100.pt') # завантажити фрейми

    agent.learn(10, batch_size_w=256, batch_size_p=256)  # запустити тренувальний цикл

    agent.save('supertrack_P_test.pt') # зберегти стан мереж
    
---

## Приклад тестування

<p align="center">
  <a href="https://github.com">
    <img src="https://github.com/user-attachments/assets/ee4c9d9a-8a7f-478c-9f7a-24ded6dbd286" alt="Logo" width="350"/>
    <img src="https://github.com/user-attachments/assets/4569896b-4327-4128-b0d6-7d8d0158ef62" alt="Logo" width="400"/>
  </a>
</p>


---

## 🧾 Ресурси що допоможуть відтворити цю роботу самостійно.

- https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
- https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
- https://theorangeduck.com/page/neural-network-not-working#init
- https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/deep_mimic
- https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html


## 🧾 Особлива подяка
Я хотів би відзначити [https://github.com/venite-xjc/SuperTrack/blob/master/README.md] за їхній цінний підхід до оптимізації навчального процесу, який дозволив досягти значних результатів у даній роботі.
