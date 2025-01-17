document.addEventListener('DOMContentLoaded', () => {
    console.log('Distributed Neural Network Framework page loaded.');

    // Example interaction for the page
    const featuresSection = document.querySelector('#features');

    if (featuresSection) {
        featuresSection.addEventListener('click', () => {
            alert('Explore the key features of the framework!');
        });
    }
});
