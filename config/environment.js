const fs = require('fs');
const rfs = require('rotating-file-stream');
const path = require('path');

const logDirectory = path.join(__dirname, '../production_logs');

fs.existsSync(logDirectory) || fs.mkdirSync(logDirectory);

const accessLogStream = rfs.createStream('access.log', {
    interval: '1d',
    path: logDirectory

});

const development = {
    name: 'development',
    asset_path: '/assets',
    session_cookie_secret: 'ihavetobefaster',
    db: 'media_website',
    smtp: {
        host: "smtp.gmail.email",
        service: "gmail",
        port: 587,
        secure: false, // true for 465, false for other ports
        auth: {
            user: '********@gmail.com', // generated ethereal user
            pass: '********' // generated ethereal password
        },
        tls: {
            rejectUnauthorized: false
        }
    },
    jwt_secret: 'Iamimproving',
    user_mail: '*************',
    morgan: {
        mode: 'dev',
        options: { stream: accessLogStream }
    }

}

const production = {
    name: 'production', 
    asset_path: process.env.CODIAL_ASSET_PATH,
    session_cookie_secret: process.env.CODIAL_SESSION_COOKIE_KEY,
    db: process.env.CODIAL_DATABASE,
    smtp: {
        host: "smtp.gmail.email",
        service: "gmail",
        port: 587,
        secure: false, // true for 465, false for other ports
        auth: {
            user: process.env.CODIAL_GMAIL_USERNAME, // generated ethereal user
            pass: process.env.CODIAL_GMAIL_PASSWORD,// generated ethereal password
        },
        tls: {
            rejectUnauthorized: false
        }
    },
    jwt_secret: process.env.CODIAL_JWT_SECRET,
    user_mail: process.env.CODIAL_GMAIL_USERNAME,
    morgan: {
        mode: 'combined',
        options: { stream: accessLogStream }
    }

}
//console.log((eval(process.env.CODIAL_ENVIRONMENT) == undefined) ? development : eval(process.env.CODIAL_ENVIRONMENT)

module.exports = development;
