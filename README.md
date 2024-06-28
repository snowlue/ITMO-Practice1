# Подготовка данных для обучения нейронных сетей
Репозиторий представляет из себя точку сбора данных для `gymnasium` со среды [Ant-v4 от MuJoCo](https://gymnasium.farama.org/environments/mujoco/ant/), которые могут пригодиться для имитационного обучения (imitation learning). Роль агента, выполняющего действия в среде, играет нейронная сеть с обучением с подкреплением (reinforcement learning).
## Как собрать данные с Ant:
0. Склонируйте репозиторий себе
```bash
git clone https://github.com/snowlue/ITMO-Practice1.git
```
### Способ №1 (простой, для тех, кто дружит с Docker)
1. Просто скопируйте и отправьте команду в терминал: (вы можете указать количество эпизодов, которое должна пройти среда, в переменной `NUM_EPISODES` — по умолчанию 100)
```bash
docker build -t ilearning . && docker run -e NUM_EPISODES=100 ilearning && docker cp $(docker ps -a --format "{{.Names}}\t{{.Image}}" | grep "ilearning" | head -n 1 | cut -f1):/app/observations.csv . && docker cp $(docker ps -a --format "{{.Names}}\t{{.Image}}" | grep "ilearning" | head -n 1 | cut -f1):/app/actions.csv .
```

### Способ №2 (для любителей делать всё самому)
1. Установите все зависимости (устанавливается облегчённый `torch` только на CPU без модулей для Nvidia):
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
2. Установите количество эпизодов, которое должна пройти среда (по умолчанию 100):  
   Для Windows: `set NUM_EPISODES=100`  
   Для Unix: `export NUM_EPISODES=100`
2. Запустите код:
```bash
python main.py
```