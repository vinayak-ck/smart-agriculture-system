const API_URL = "http://127.0.0.1:8000/api/sensor/get/";

async function fetchData() {
    let res = await fetch(API_URL);
    let data = await res.json();

    let table = document.querySelector("#sensorTable tbody");
    table.innerHTML = "";

    data.forEach(item => {
        let row = `
        <tr>
            <td>${item.temperature}</td>
            <td>${item.humidity}</td>
            <td>${item.ph}</td>
            <td>${item.nitrogen}</td>
            <td>${item.phosphorus}</td>
            <td>${item.potassium}</td>
            <td>${item.created_at}</td>
        </tr>`;
        table.innerHTML += row;
    });

    updateChart(data);
}

setInterval(fetchData, 3000);

fetchData();


// 🔹 Chart
let chart;

function updateChart(data) {
    let labels = data.map(d => d.created_at);
    let values = data.map(d => d.temperature); // example

    if (chart) chart.destroy();

    chart = new Chart(document.getElementById("growthChart"), {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Temperature",
                data: values
            }]
        }
    });
}


// 🔹 Image Upload (dummy for now)
function uploadImage() {
    document.getElementById("result").innerText = "Prediction coming soon...";
}