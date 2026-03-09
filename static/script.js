// ===== Utils =====
function getClientId() {
    let id = localStorage.getItem("client_id");
    if (!id) {
        id = crypto.randomUUID();
        localStorage.setItem("client_id", id);
    }
    return id;
}

function setupPreview(input, img) {
    input.addEventListener("change", () => {
        if (input.files?.[0]) {
            img.src = URL.createObjectURL(input.files[0]);
            img.style.display = "block";
        }
    });
}

function setLoading(isLoading) {
    spinner.style.display = isLoading ? "flex" : "none";
    tryonBtn.disabled = isLoading;
}

// ===== DOM =====
const clientId = getClientId();

const els = {
    personInput: document.getElementById("personInput"),
    clothInput: document.getElementById("clothInput"),
    personPreview: document.getElementById("personPreview"),
    clothPreview: document.getElementById("clothPreview"),
    tryonBtn: document.getElementById("tryonBtn"),
    spinner: document.getElementById("loadingSpinner"),
    resultImg: document.getElementById("resultImage"),
    segImg: document.getElementById("segmentationImage"),
    resultActions: document.getElementById("resultActions"),
    downloadBtn: document.getElementById("downloadBtn"),
    resetBtn: document.getElementById("resetBtn")
};

const {
    personInput, clothInput, personPreview, clothPreview,
    tryonBtn, spinner, resultImg, segImg, resultActions,
    downloadBtn, resetBtn
} = els;

// ===== Preview =====
setupPreview(personInput, personPreview);
setupPreview(clothInput, clothPreview);

// ===== Download =====
downloadBtn.addEventListener("click", () => {
    const url = resultImg.dataset.downloadUrl;
    if (!url) return;

    const a = document.createElement("a");
    a.href = url;
    a.download = "tryon_result.png";
    a.click();
});

// ===== Reset =====
resetBtn.addEventListener("click", () => {
    segImg.style.display = "none";
    resultImg.style.display = "none";
    resultActions.style.display = "none";
});

// ===== Try-On =====
tryonBtn.addEventListener("click", async () => {
    const person = personInput.files[0];
    const cloth  = clothInput.files[0];

    if (!person || !cloth) {
        alert("Vui lòng chọn đủ ảnh người và quần áo!");
        return;
    }

    setLoading(true);
    resultImg.style.display = "none";
    segImg.style.display = "none";

    const formData = new FormData();
    formData.append("person", person);
    formData.append("cloth", cloth);
    formData.append("client_id", clientId);

    try {
        const res = await fetch("/tryon", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        // Segmentation
        if (data.segmentation_url) {
            segImg.src = data.segmentation_url + "?t=" + Date.now();
            segImg.style.display = "block";
        }

        // Result
        resultImg.src = data.result_url + "?t=" + Date.now();
        resultImg.onload = () => {
            setLoading(false);
            resultImg.style.display = "block";
            resultActions.style.display = "flex";
            resultImg.dataset.downloadUrl = data.result_url;
        };

    } catch (err) {
        console.error(err);
        setLoading(false);
        alert("Có lỗi khi chạy Try-On!");
    }
});
