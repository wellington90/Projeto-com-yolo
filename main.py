# CARREGA AS DEPENDENCIAS
import cv2
import time

#CORES DAS CLASSES
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# CARREGA AS CLASSES
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# CAPIURA DO VIDEO
cap = cv2.VideoCapture("01.mp4")

# CARREGANDO OS PESOS DA REDF NEURAL
#net = cv2.dnn.readNnet ("weights/yolova4.weights", "“cfg/yolova4.cfg")
net = cv2.dnn.readNet("yolova-tiny.weights", "yolova-tiny.cfg")

# SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel (net)
model . setInputParams (size=(416, 416), scale=1/255)


#LENDO OS FRAMES DO VIDEO
while True:

    # CAPTURA DO FRAME
    _, frame = cap.read()

    # COMEÇO DA CONTAGEM DOS MS
    start = time.time()

    # DETECÇÃO
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # PROCESSA AS DETECÇÕES, SE TIVER ALGUMA
    if len(classes) > 0:
        # PERCORRER TODAS AS DETECÇÕES
        for (classid, score, box) in zip(classes, scores, boxes):

            # GERANDO UMA COR PARA A CLASSE
            color = COLORS[int(classid) % len(COLORS)]

            # PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
            label = f'{class_names[classid]} : {score}'

            # DESENHANDO A BOX DA DETECÇÃO
            cv2.rectangle(frame, box, color, 2)

            # ESCREVENDO O NOME DA CLASSE EM CIMA DA BOX DO OBJETO
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # MOSTRAR O FRAME NA TELA
    cv2.imshow("YOLO Object Detection", frame)

    # SAIR COM A TECLA "Q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# LIBERAR OS RECURSOS
cap.release()
cv2.destroyAllWindows()
