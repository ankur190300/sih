const Job = require('../models/jobs')

module.exports.create = function(req, res){
    return res.render('job',
            {
                title: 'Job',
                
            });
};

module.exports.create_job = async function(req, res){

    try{

        let p= await Job.create({

            description: req.body.description,
            company: req.user._id,
            job_name: req.body.job_name, 
            salary: req.body.salary

         })

         req.flash('success', "Job created!")
        return res.redirect('back');

    }catch(err){
        req.flash('error', err);
        
        return res.redirect('back');
    }

}