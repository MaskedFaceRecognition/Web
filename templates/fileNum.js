// function createWindow () { 
//     let win = new BrowserWindow({ 
//         width: 800, height: 600, 
//         webPreferences: { 
//             nodeIntegration: true, 
//             contextIsolation: false, //enableRemoteModule: true, 
//         } }); //개발자 모드 (F12) 
//         win.webContents.openDevTools(); 
//         win.loadFile('index.html'); 
//     }

function createAddItemWindow() {
    //Create new window
    addItemWindown = new BrowserWindow({
        width: 300,
        height: 200,
        title: 'Add Item',

        //The lines below solved the issue
        webPreferences: {
            nodeIntegration: true
        }
    })
}



function angryNum(){
    const fs = require('fs');
    const dir = 'angry';

    fs.readdir(dir, (err, files) => {
        console.log(files.length);
        document.write(files.length);
    });
}

function happyNum(){
    const fs = require('fs');
    const dir = 'happy';
    var count = 0

    fs.readdir(dir, (err, files) => {
        console.log(files.length);
        document.write(files.length);
    });
}

function neutralNum(){
    const fs = require('fs');
    const dir = 'neutral';

    fs.readdir(dir, (err, files) => {
        console.log(files.length);
        document.write(files.length);
    });
}