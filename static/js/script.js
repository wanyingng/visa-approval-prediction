document.getElementById("visaForm").addEventListener("submit", function (event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const prediction = document.getElementById("prediction");

    // Clear previous prediction
    prediction.textContent = "???";

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            // Show the latest prediction
            prediction.textContent = data.context;
        })
        .catch(error => {
            console.error("Error:", error);
            // Display error message
            prediction.textContent = "An error occurred.";
        });
});
