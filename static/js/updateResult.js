async function updateResult() {
    try {
        const response = await fetch('/result');
        const data = await response.json();
        document.getElementById('result-text').textContent = data.result;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result-text').textContent = 'Error loading result';
    }
}

updateResult();
setInterval(updateResult, 1000);