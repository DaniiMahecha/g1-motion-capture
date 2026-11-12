import cv2  # OpenCV: librería para capturar, procesar y mostrar video

# VideoCapture abre una fuente de video y devuelve un objeto para "pedirle" frames.
# 0 = cámara por defecto del computador. Si no abre, prueba con 1.
cap = cv2.VideoCapture(0)

# isOpened() devuelve True si la cámara se abrió bien.
# Si falla (cámara ocupada, índice incorrecto, sin permisos), avisamos y salimos.
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Un video es una secuencia de imágenes (frames). Este loop repite
# "leer frame -> mostrar frame" muy rápido, y eso da la sensación de video en vivo.
while True:
    # TODO 1: cap.read() devuelve DOS valores:
    #   success -> True si logró leer el frame, False si falló
    #   frame   -> la imagen capturada (un array de NumPy en formato BGR)
    success, frame = cap.read()

    # TODO 2: si la lectura falló, frame no sirve (puede ser None y
    # cv2.imshow daría error), así que avisamos y rompemos el loop.
    if not success:
        print("No se pudo leer el frame de la cámara")
        break

    # TODO 3: abre (o actualiza) una ventana llamada "Demo tracking"
    # y dibuja ahí el frame actual. Se llama una vez por cada frame.
    cv2.imshow("Demo tracking", frame)

    # TODO 4: waitKey(1) espera 1 ms a que se presione una tecla.
    # Sirve para dos cosas: (a) deja que OpenCV refresque la ventana
    # (sin esto la ventana se congela) y (b) detecta teclas.
    # Devuelve el código de la tecla; "& 0xFF" se queda con el último byte
    # para que la comparación funcione igual en cualquier sistema.
    # ord('q') convierte el carácter 'q' a su código numérico.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # si presionaste 'q', salimos del loop

# TODO 5: limpieza al terminar.
cap.release()            # libera la cámara para que otros programas puedan usarla
cv2.destroyAllWindows()  # cierra todas las ventanas de OpenCV