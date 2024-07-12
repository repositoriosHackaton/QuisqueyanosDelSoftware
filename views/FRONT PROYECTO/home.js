// home.js
document.addEventListener('DOMContentLoaded', function() {
    const chatboxButton = document.querySelector('.chatbox__button button');
    const chatboxSupport = document.querySelector('.chatbox__support');
    
    chatboxButton.addEventListener('click', function() {
        chatboxSupport.classList.toggle('chatbox--active');
    });
});
