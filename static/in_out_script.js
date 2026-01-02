const video = document.getElementById("video");
const msg = document.getElementById("message");

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

async function captureAndSend(){
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async blob => {
        const formData = new FormData();
        formData.append("image", blob, "frame.jpg");

        let res = await fetch("http://127.0.0.1:5000/mark_attendance", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        msg.innerText = data.message;
    }, "image/jpeg");
}

setInterval(captureAndSend, 2000);
