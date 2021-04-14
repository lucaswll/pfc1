const cam = document.getElementById('cam')

const startVideo = () => {
    navigator.mediaDevices.enumerateDevices()
    .then(devices => {
        if(Array.isArray(devices)) {
            devices.forEach(device => {
                if(device.kind == 'videoinput'){
                    if(device.label.includes('')){ //parte do label da cam, caso tenha mais de 1
                        navigator.getUserMedia(
                            {video: {
                                deviceId: device.deviceId
                            }},
                            stream => cam.srcObject = stream,
                            error => console.log(error)
                        )
                    }                    
                }
            })
        }
    })
}

const loadLabels = () => {
    const labels = ['LucasWallace']

    return Promise.all(labels.map(async label => {
        const descriptions = []

        for (let i = 1;i <= 4; i++){
            const img = await faceapi.fetchImage(`/lib/labels/${label}/lucas${i}.jpg`)

            const detections = await faceapi
                .detectSingleFace(img)
                .withFaceLandmarks()
                .withFaceDescriptor()
            descriptions.push(detections.descriptor)
        }
        return new faceapi.LabeledFaceDescriptors(label, descriptions) //pro LucasWallace(label), tenho as descrições (usadas no programa, so n sei oq)
    }))
}

//redes neurais do face-api
Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/lib/models'), //detecta e desenha o quadrado ao redor do rosto
    faceapi.nets.faceLandmark68Net.loadFromUri('/lib/models'), //reconhece e desenha os traços do rosto
    faceapi.nets.faceRecognitionNet.loadFromUri('/lib/models'), //conhecimento do rosto - reconhecimento facial
    faceapi.nets.faceExpressionNet.loadFromUri('/lib/models'), //detecta expressão facial
    faceapi.nets.ageGenderNet.loadFromUri('/lib/models'), //detecta idade e gênero
    faceapi.nets.ssdMobilenetv1.loadFromUri('/lib/models') //detectar o rosto (debaixo dos panos)
]).then(startVideo)

cam.addEventListener('play', async() => {
    const canvas = faceapi.createCanvasFromMedia(cam)
    const canvasSize = {
        width: cam.width,
        height: cam.height
    }

    const labels = await loadLabels()

    faceapi.matchDimensions(canvas, canvasSize)
    document.body.appendChild(canvas)

    setInterval(async() => {
        const detections = await faceapi
            .detectAllFaces(cam, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceExpressions() //retirar
            .withAgeAndGender() //retrirar
            .withFaceDescriptors()

        const resizedDetections = faceapi.resizeResults(detections, canvasSize)
        //console.log(detections) //qnd detecta uma face, ele retorna o x, y, altura, largura, pra montar o quadrado
        const faceMatcher = new faceapi.FaceMatcher(labels, 0.7)
        const results = resizedDetections.map(d =>
            faceMatcher.findBestMatch(d.descriptor))

        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height) //limpa o canvas (quadrado) limpando de x=y=0 até a posicao do width do quadrado 
        
        faceapi.draw.drawDetections(canvas, resizedDetections) //pedindo pra desenhar o canvas
                //1º param: onde quero que desenhe , 2º: fonte das infos.
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections) //pedindo pra desenhar os landmarks

        faceapi.draw.drawFaceExpressions(canvas, resizedDetections)

        resizedDetections.forEach(detection => {
            const {age, gender, genderProbability} = detection

            new faceapi.draw.DrawTextField([
                `${parseInt(age, 10)} anos`,
                `${gender} (${parseInt(genderProbability*100, 10)})`
            ], detection.detection.box.topRight).draw(canvas)            
        })

        results.forEach((result, index) => {
            const box = resizedDetections[index].detection.box
            const {label, distance} = result

            new faceapi.draw.DrawTextField([
                `${label} (${parseInt(distance*100, 10)})`
            ], box.bottomRight).draw(canvas)
        })

    }, 100)
})