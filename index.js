const express = require("express");
const app = express();
require('./config/view-helper')(app);
const port = 8000;
const expressLayouts = require("express-ejs-layouts");
const db = require("./config/mongoose");
const cookieParser = require("cookie-parser");
const session = require("express-session");
const passport = require("passport");
const passportLocal = require("./config/passport-local-strategy");
const passportJWT = require("./config/passport-jwt-strategy");
const MongoStore = require('connect-mongo')(session);
const sassMiddleware = require('node-sass-middleware');
const flash = require('connect-flash');
const customWare = require("./config/middleware")
const chatServer = require('http').Server(app);
const chatSockets = require('./config/chat_sockets.js').chatSockets(chatServer);
const env = require('./config/environment');
const logger = require('morgan');
const path = require('path');

chatServer.listen(5000);
console.log("Chat server is running on port 5000");

if (env.name == 'development') {
    app.use(sassMiddleware({
        src: path.join(__dirname, env.asset_path, 'scss'),
        dest: path.join(__dirname, env.asset_path, 'css'),
        debug: true,
        outputStyle: 'extended',
        prefix: '/css'

    }));


}

//reading the post request from the form 
app.use(express.urlencoded());

//using the cookieparse as a middleware
app.use(cookieParser());


//adding the static files to the project

app.use(express.static('./assets'));


app.use('/uploads', express.static(__dirname + '/uploads'));

app.use(logger(env.morgan.mode, env.morgan.options));

// includeing the layout variable for the backend layouts 
//always remember to include it before using the router
app.use(expressLayouts);



// extract css and javascript files from the base files to the layout
app.set("layout extractStyles", true);
app.set("layout extractScripts", true);


//setting up the view engine

app.set('view engine', 'ejs');
app.set('views', './views');


//for cookies
app.use(session({

    name: 'social_media_website',
    secret: env.session_cookie_secret,
    saveUninitialized: false,
    resave: false,
    cookie: {
        maxAge: (1000 * 60 * 100)
    },
    store: new MongoStore({

        autoRemove: 'disabled',
        mongooseConnection: db
    }, function (err) {
            console.log(err||"connect-mongo status ok ")
    }
        )


}));

app.use(passport.initialize());
app.use(passport.session());




app.use(flash());
app.use(customWare.setflash);

app.use(passport.setAuthenticatedUser);


//use express router
app.use('/', require('./routes'));

app.listen(port, function (err) {

    if (err) {
        console.log(`unable to connect to the server due to error ${err}`);
        return;

    }
    else
        console.log(" express server is up and running on port", port);
});