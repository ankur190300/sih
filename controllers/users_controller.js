
const User = require('../models/JavaScript1');
const fs = require('fs');
const path = require('path');



// render the user profile page 
module.exports.profile = function (req, res) {

    User.findById(req.params.id, function (err, user) {
        return res.render('profile',
            {
                title: 'profile',
                profile_user:user
            });

    })

   
};

module.exports.update = async function (req, res) {

    if (req.params.id == req.user.id) {
        try {/*

            User.findByIdAndUpdate(req.params.id, { name: req.body.name, email: req.body.email }, function (err, user) {
                req.flash('success', "Updated")
                res.redirect('back');


            })*/
            
            let user = await User.findById(req.params.id);
            User.uploadAvatar(req, res, function (err) {
                if (err) {
                    console.log("**********Multer Error", err);
                }
                
                user.name = req.body.name;
                user.email = req.body.email;

                if (req.file) {

                    // delete the old avatar
                    if (user.avatar) {
                        //checking if file exists
                        if (fs.existsSync(path.join(__dirname, '..', user.avatar))) {
                            fs.unlinkSync(path.join(__dirname, '..', user.avatar));
                        }
                       
                    }

                    // add new path 
                    user.avatar = User.avatarPath + '/' + req.file.filename;
                }
                user.save();
                return res.redirect('back');

            });
            
        }
        catch(err){
            req.flash('error', err);
            console.log(err);
            return res.redirect('back');
        }

    }
    else {
        
        req.flash('error', 'Unauthorized !')
        res.status(401).send('Unauthorized');
    }
}


//render the sign in page 
module.exports.signin = function (req, res) {

    if (req.isAuthenticated()) {
        return res.redirect('/users/profile');
    }

    return res.render('user_sign_in', {
        title: "user sign-in"
    });
};
 
//render the sign up page 
module.exports.signup = function (req, res) {

    if (req.isAuthenticated()) {
        return res.redirect('/users/profile');
    }


    return res.render('user_sign_up', {
        title: "user sign-up"
    });
};
//set up the user id create page 
module.exports.create = function (req, res) {
    if (req.body.password != req.body.confirm_password) {
        console.log("the password and confirm password are not matching");
        return res.redirect("back");
    };
    User.findOne({ email: req.body.email }, function (err, user) {

        if (err) {
            console.log("there was an error in finding user in the database");
            return;
        }

        if (!user) {
            User.create(req.body, function (err, user) {

                if (err) {
                    console.log("there was an error in creating user in the database");
                    return;
                }
                return res.redirect("/users/sign-in");
            });
        }
        else {
            console.log("the account with this id is already set up");
            return res.redirect("back");
        }
    });
    
};

module.exports.createSession = function (req, res) {
    req.flash('success', 'Logged in successfully');

    return res.redirect('/');
};

module.exports.destroySession = function (req, res) {

    req.logout();
    req.flash('success', 'You have logged out');
    return res.redirect('/');
}