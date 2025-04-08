async function processData() {
    const loadingContainer = document.querySelector('.loading-container');
    const resultDiv = document.getElementById('result');
    
    try {
        // Show loading animation first
        loadingContainer.style.display = 'flex';
        
        // fetch returns a Promise, await will wait for this Promise to resolve
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ /* your data */ })
        });
        
        // response.json() also returns a Promise, so we await it too
        const data = await response.json();
        
        // This line won't execute until the data is received
        resultDiv.textContent = data.result;
    } catch (error) {
        resultDiv.textContent = 'Error occurred';
    } finally {
        // This will run after everything is done (or if an error occurs)
        loadingContainer.style.display = 'none';
    }
}
