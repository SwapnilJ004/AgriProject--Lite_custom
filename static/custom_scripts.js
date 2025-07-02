console.log("In application");
let drop_down_item = document.querySelector('#dropdown');
let input_container = document.querySelector('.file-input-container');

console.log(drop_down_item);
drop_down_item.addEventListener('click', () => {
    console.log('click');
    input_container.classList.remove('hide');
});